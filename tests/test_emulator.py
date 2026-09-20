import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from src import config
from src.params import align_to_params
from src.emulator import (
    default_model,
    combined_features,
    cross_val_predict_emulator,
    train_full_model,
    prediction_metrics,
    null_control_metrics,
    feature_importance,
)


def _fast_model(random_state=42):
    """A small forest for tests -- default_model's 500 trees would make
    the test suite (which fits many models across CV folds and
    null-control shuffles) unusably slow; the statistical behavior this
    module's tests care about doesn't need that many trees to show up."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=10, random_state=random_state, n_jobs=1)


def _synthetic_case(seed=0, n_sims=300, n_bins=20, noise=0.01):
    """
    theta with the real LH ranges for Omega_m/sigma_8; X mostly noise
    columns, column 0 a clean *linear* function of Omega_m, column 1 a
    clean *nonlinear* function of sigma_8 -- the nonlinear column is why
    a random forest should beat a plain linear baseline.
    """
    rng = np.random.default_rng(seed)
    theta = pd.DataFrame({
        "Omega_m": rng.uniform(0.1, 0.5, n_sims),
        "sigma_8": rng.uniform(0.6, 1.0, n_sims),
    })
    X = rng.normal(0, 1, (n_sims, n_bins))
    X[:, 0] = theta["Omega_m"].values + rng.normal(0, noise, n_sims)
    X[:, 1] = theta["sigma_8"].values ** 2 + rng.normal(0, noise, n_sims)
    y = theta[["Omega_m", "sigma_8"]].values
    return X, y, theta


# ---------------------------------------------------------------------
# Alignment / no-leakage correctness
# ---------------------------------------------------------------------

def test_combined_features_uses_align_common_sims_and_concatenates():
    sim_ids_agn = np.array([3, 1, 5, 2])
    residuals_agn = np.array([[30.0], [10.0], [50.0], [20.0]])
    sim_ids_gal = np.array([2, 5, 9])
    residuals_gal = np.array([[200.0], [500.0], [900.0]])

    common, X = combined_features(sim_ids_agn, residuals_agn, sim_ids_gal, residuals_gal)

    assert list(common) == [2, 5]
    assert X.shape == (2, 2)
    # row 0 -> sim 2: agn=20, gal=200 ; row 1 -> sim 5: agn=50, gal=500
    assert np.array_equal(X, [[20.0, 200.0], [50.0, 500.0]])


def test_cross_val_predict_emulator_rejects_misaligned_lengths():
    X = np.zeros((10, 3))
    y = np.zeros(9)
    with pytest.raises(ValueError):
        cross_val_predict_emulator(X, y)


def test_cross_val_predict_emulator_rejects_too_many_splits():
    X = np.zeros((5, 3))
    y = np.zeros(5)
    with pytest.raises(ValueError):
        cross_val_predict_emulator(X, y, n_splits=10)


def test_cross_val_predict_emulator_gives_genuinely_held_out_predictions():
    # A 1-nearest-neighbor model that (incorrectly) saw its own query row
    # during "training" reproduces that row's y EXACTLY (distance-0
    # self-match). On continuous random data an exact match is
    # astronomically improbable unless a row leaked into its own fold.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = rng.normal(size=200)

    y_pred = cross_val_predict_emulator(
        X, y, model_factory=lambda random_state=None: KNeighborsRegressor(n_neighbors=1),
        n_splits=5,
    )

    assert not np.allclose(y_pred, y, atol=1e-10)


# ---------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------

def test_recovers_known_linear_and_nonlinear_signal():
    X, y, _ = _synthetic_case()
    y_pred = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5)
    metrics = prediction_metrics(y, y_pred, target_names=["Omega_m", "sigma_8"])

    assert (metrics["r2"] > 0.8).all(), metrics


def test_random_forest_beats_linear_regression_on_a_nonlinear_signal():
    # An XOR-style interaction: y depends on the SIGN combination of two
    # features, not their magnitude -- by construction y is linearly
    # uncorrelated with either feature alone (a linear model can't do
    # better than predicting the mean), but a tree-based model learns the
    # combination easily. This is the concrete case for choosing a random
    # forest over a linear baseline, not just an assertion in a docstring.
    rng = np.random.default_rng(5)
    n = 400
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = ((f1 > 0) ^ (f2 > 0)).astype(float)
    X = np.column_stack([f1, f2, rng.normal(size=(n, 8))])

    rf_pred = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5)
    lin_pred = cross_val_predict_emulator(
        X, y, model_factory=lambda random_state=None: LinearRegression(), n_splits=5,
    )

    rf_r2 = prediction_metrics(y, rf_pred, target_names=["xor"])["r2"].iloc[0]
    lin_r2 = prediction_metrics(y, lin_pred, target_names=["xor"])["r2"].iloc[0]
    assert rf_r2 > lin_r2 + 0.3
    assert lin_r2 < 0.1  # confirms the fixture really is linearly uncorrelated


# ---------------------------------------------------------------------
# Negative/null control -- the single most important check
# ---------------------------------------------------------------------

def test_null_control_correctly_flags_pure_noise_as_not_predictive():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 10))
    y = rng.normal(size=(150, 2))

    y_pred = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5)
    obs = prediction_metrics(y, y_pred, target_names=["a", "b"])
    null = null_control_metrics(X, y, target_names=["a", "b"], model_factory=_fast_model, n_shuffles=20, n_splits=5)

    for target in ["a", "b"]:
        obs_r2 = obs.set_index("target").loc[target, "r2"]
        null_r2 = null[null["target"] == target]["r2"]
        assert obs_r2 <= np.percentile(null_r2, 95) + 0.1


def test_null_control_floor_is_far_below_a_real_signal():
    X, y, _ = _synthetic_case(n_sims=200)
    target_names = ["Omega_m", "sigma_8"]

    y_pred = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5)
    obs = prediction_metrics(y, y_pred, target_names=target_names)
    null = null_control_metrics(X, y, target_names=target_names, model_factory=_fast_model, n_shuffles=20, n_splits=5)

    for target in target_names:
        obs_r2 = obs.set_index("target").loc[target, "r2"]
        obs_rmse = obs.set_index("target").loc[target, "rmse"]
        null_r2 = null[null["target"] == target]["r2"]
        null_rmse = null[null["target"] == target]["rmse"]
        assert obs_r2 > np.percentile(null_r2, 95)
        assert obs_rmse < np.percentile(null_rmse, 5)


# ---------------------------------------------------------------------
# Metrics-function correctness
# ---------------------------------------------------------------------

def test_prediction_metrics_perfect_predictions_gives_r2_one_rmse_zero():
    y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    metrics = prediction_metrics(y, y, target_names=["a", "b"])
    assert np.allclose(metrics["r2"], 1.0)
    assert np.allclose(metrics["rmse"], 0.0)


def test_prediction_metrics_shape_mismatch_raises():
    y_true = np.zeros((10, 2))
    y_pred = np.zeros((10, 3))
    with pytest.raises(ValueError):
        prediction_metrics(y_true, y_pred, target_names=["a", "b", "c"])


def test_prediction_metrics_rejects_zero_variance_target():
    y_true = np.column_stack([np.ones(10), np.arange(10)])
    y_pred = np.column_stack([np.ones(10), np.arange(10)])
    with pytest.raises(ValueError):
        prediction_metrics(y_true, y_pred, target_names=["constant", "varying"])


def test_prediction_metrics_matches_manual_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5, 5.5])

    metrics = prediction_metrics(y_true, y_pred, target_names=["x"])

    manual_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    manual_r2 = 1 - ss_res / ss_tot

    assert metrics["rmse"].iloc[0] == pytest.approx(manual_rmse)
    assert metrics["r2"].iloc[0] == pytest.approx(manual_r2)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def test_cross_val_predict_emulator_is_reproducible_with_same_random_state():
    X, y, _ = _synthetic_case(n_sims=100)
    pred1 = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5, random_state=7)
    pred2 = cross_val_predict_emulator(X, y, model_factory=_fast_model, n_splits=5, random_state=7)
    assert np.array_equal(pred1, pred2)


def test_train_full_model_is_reproducible_with_same_random_state():
    X, y, _ = _synthetic_case(n_sims=100)
    m1 = train_full_model(X, y, model_factory=_fast_model, random_state=7)
    m2 = train_full_model(X, y, model_factory=_fast_model, random_state=7)
    assert np.array_equal(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------
# Wiring / subtlety tests
# ---------------------------------------------------------------------

def test_multioutput_call_matches_independent_per_target_calls_under_scale_mismatch():
    # sklearn's multi-output tree splitting criterion is evaluated jointly
    # across targets, so a naive joint fit lets a much-higher-variance
    # target dominate split choices and badly degrade a lower-variance
    # one -- confirmed directly: with the mismatched scales below, a raw
    # joint RandomForestRegressor fit drops target 'a's R^2 from ~0.99 to
    # negative. cross_val_predict_emulator avoids this by fitting each
    # target independently (see its docstring), so calling it with a
    # 2-column y must give (near-)exactly the same predictions as calling
    # it once per column -- not just "not badly worse".
    rng = np.random.default_rng(3)
    n_sims, n_bins = 200, 10
    X = rng.normal(size=(n_sims, n_bins))
    a = X[:, 0] + rng.normal(0, 0.05, n_sims)          # ~ O(1)
    b = 1000 * X[:, 1] + rng.normal(0, 5, n_sims)       # ~ O(1000)
    y_joint = np.column_stack([a, b])

    # sanity: a genuinely naive joint fit (not this module's function)
    # really does suffer the scale-domination bug on this fixture --
    # otherwise this test wouldn't be exercising a real phenomenon.
    from sklearn.model_selection import KFold, cross_val_predict as _cvp
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    naive_joint_pred = _cvp(_fast_model(random_state=42), X, y_joint, cv=cv)
    naive_a_r2 = prediction_metrics(a, naive_joint_pred[:, 0], target_names=["a"])["r2"].iloc[0]
    assert naive_a_r2 < 0.5, "fixture no longer demonstrates the scale-domination bug"

    multi_pred = cross_val_predict_emulator(X, y_joint, model_factory=_fast_model, n_splits=5)
    a_pred = cross_val_predict_emulator(X, a, model_factory=_fast_model, n_splits=5)
    b_pred = cross_val_predict_emulator(X, b, model_factory=_fast_model, n_splits=5)

    assert np.array_equal(multi_pred[:, 0], a_pred)
    assert np.array_equal(multi_pred[:, 1], b_pred)


def test_feature_importance_shape():
    X, y, _ = _synthetic_case(n_sims=100, n_bins=12)
    model = train_full_model(X, y, model_factory=_fast_model)
    importance = feature_importance(model, n_k=3, n_r=4)
    assert importance.shape == (3, 4)
    assert np.isclose(importance.sum(), 1.0)  # RF importances sum to 1


def test_feature_importance_rejects_non_tree_model():
    X, y, _ = _synthetic_case(n_sims=50, n_bins=6)
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError):
        feature_importance(model, n_k=2, n_r=3)


def test_feature_importance_rejects_wrong_bin_count():
    X, y, _ = _synthetic_case(n_sims=100, n_bins=12)
    model = train_full_model(X, y, model_factory=_fast_model)
    with pytest.raises(ValueError):
        feature_importance(model, n_k=2, n_r=3)  # 6 != 12


def test_default_model_has_documented_parameters():
    # Checks the constructed (unfitted) estimator's parameters directly --
    # never fits a 500-tree forest in the test suite, just confirms the
    # documented default is what's actually configured.
    model = default_model(random_state=7)
    assert model.n_estimators == 500
    assert model.random_state == 7
    assert model.n_jobs == -1


# ---------------------------------------------------------------------
# Integration test: full chain against fake but realistically-shaped data
# ---------------------------------------------------------------------

def test_full_pipeline_on_fake_lh_and_galaxy_shaped_data():
    # Deliberately small (not the real ~1000-sim, 150-bin scale) -- this
    # test exists to exercise the real wiring (alignment, shapes, the
    # full call chain), not to check statistical behavior, which the
    # dedicated positive/negative-control tests above already cover at a
    # size chosen for that purpose.
    rng = np.random.default_rng(42)
    n_k, n_r = 3, 10
    n_bins = n_k * n_r
    n_pool = 150

    all_ids = np.arange(n_pool)
    theta_all = pd.DataFrame(
        {p: rng.uniform(0, 1, n_pool) for p in config.ALL_PARAMS}, index=all_ids,
    )
    theta_all.index.name = "sim_id"
    theta_all.loc[:, "Omega_m"] = rng.uniform(0.1, 0.5, n_pool)
    theta_all.loc[:, "sigma_8"] = rng.uniform(0.6, 1.0, n_pool)

    # AGN retains a random subset, galaxy a different (smaller) subset --
    # fixed-N retention drops different sims per tracer in real data too.
    agn_ids = np.sort(rng.choice(all_ids, size=120, replace=False))
    gal_ids = np.sort(rng.choice(all_ids, size=100, replace=False))

    agn_theta = align_to_params(agn_ids, theta_all)
    agn_residuals = (
        agn_theta["Omega_m"].values[:, None] * rng.normal(1, 0.1, n_bins)[None, :]
        + rng.normal(0, 0.05, size=(len(agn_ids), n_bins))
    )

    gal_theta = align_to_params(gal_ids, theta_all)
    gal_residuals = (
        gal_theta["sigma_8"].values[:, None] * rng.normal(1, 0.1, n_bins)[None, :]
        + rng.normal(0, 0.05, size=(len(gal_ids), n_bins))
    )

    # AGN-only and galaxy-only
    agn_pred = cross_val_predict_emulator(
        agn_residuals, agn_theta[["Omega_m", "sigma_8"]].values,
        model_factory=_fast_model, n_splits=5,
    )
    agn_metrics = prediction_metrics(
        agn_theta[["Omega_m", "sigma_8"]].values, agn_pred, target_names=["Omega_m", "sigma_8"],
    )
    assert set(agn_metrics["target"]) == {"Omega_m", "sigma_8"}

    # combined features
    common_ids, X_combined = combined_features(agn_ids, agn_residuals, gal_ids, gal_residuals)
    assert X_combined.shape == (len(common_ids), 2 * n_bins)
    combined_theta = align_to_params(common_ids, theta_all)
    combined_pred = cross_val_predict_emulator(
        X_combined, combined_theta[["Omega_m", "sigma_8"]].values,
        model_factory=_fast_model, n_splits=5,
    )
    combined_metrics = prediction_metrics(
        combined_theta[["Omega_m", "sigma_8"]].values, combined_pred,
        target_names=["Omega_m", "sigma_8"],
    )
    assert len(combined_metrics) == 2

    # null control (tiny n_shuffles -- this is a wiring check, not a
    # statistical one; test_null_control_* above cover the statistics)
    null = null_control_metrics(
        agn_residuals, agn_theta[["Omega_m", "sigma_8"]].values,
        target_names=["Omega_m", "sigma_8"], model_factory=_fast_model, n_shuffles=3, n_splits=3,
    )
    assert set(null["target"]) == {"Omega_m", "sigma_8"}
    assert len(null) == 2 * 3

    # feature importance
    model = train_full_model(
        agn_residuals, agn_theta[["Omega_m", "sigma_8"]].values, model_factory=_fast_model,
    )
    importance = feature_importance(model, n_k, n_r)
    assert importance.shape == (n_k, n_r)
