import numpy as np
import pandas as pd
import pytest

from src.onep import (
    parse_1p_label,
    infer_1p_parameter_names,
    group_steps,
    mean_cdf_by_step,
    monotonic_trend,
    step_label,
    fiducial_value,
)


def test_parse_1p_label():
    assert parse_1p_label("p1_3") == (1, 3)
    assert parse_1p_label("p6_0") == (6, 0)
    assert parse_1p_label("p1_n2") == (1, -2)
    assert parse_1p_label("p28_n1") == (28, -1)


def test_parse_1p_label_double_digit_index():
    # regression: a naive single-digit regex would silently mis-parse or
    # reject this -- caught by a real error report against actual data
    assert parse_1p_label("p10_1") == (10, 1)
    assert parse_1p_label("p10_n1") == (10, -1)


def test_parse_1p_label_fiducial():
    # the shared fiducial belongs to every parameter, not one of its own
    assert parse_1p_label("0") == (None, 0)


def test_parse_1p_label_rejects_bad_format():
    for bad in ["LH_1", "1_3", "1", "p1_", "abc", "p1_2_3", "p1_n", "p1"]:
        with pytest.raises(ValueError):
            parse_1p_label(bad)


def _synthetic_1p_theta(all_params, steps=(-2, -1, 1, 2), seed=0):
    """
    One shared fiducial ("0") plus, for each parameter, `steps` rows where
    exactly that column is offset from fiducial and the rest sit at it --
    mirrors the real structure (single shared fiducial reused across every
    parameter's sweep, not a per-group fiducial).
    """
    rng = np.random.default_rng(seed)
    fiducial = {p: rng.uniform(0.3, 0.7) for p in all_params}

    rows = {"0": dict(fiducial)}
    for pidx, target in enumerate(all_params, start=1):
        for step in steps:
            label = f"p{pidx}_{step}" if step >= 0 else f"p{pidx}_n{abs(step)}"
            row = dict(fiducial)
            row[target] = fiducial[target] + step * 0.05
            rows[label] = row

    return pd.DataFrame(rows).T[all_params]


def test_infer_1p_parameter_names_discovers_from_columns_not_a_fixed_list():
    # deliberately NOT config.ALL_PARAMS -- proves the function doesn't
    # depend on that list at all, matching a 1P set with its own
    # parameterization (e.g. CAMELS's 28-astrophysics-parameter 1P set)
    params = ["Omega0", "sigma8", "WindEnergyIn1e51erg", "RadioFeedbackFactor"]
    theta_1p = _synthetic_1p_theta(params)

    mapping, ambiguous = infer_1p_parameter_names(theta_1p)

    assert ambiguous == {}
    assert mapping == {i + 1: p for i, p in enumerate(params)}


def test_infer_1p_parameter_names_excludes_seed():
    params = ["Omega0", "sigma8"]
    theta_1p = _synthetic_1p_theta(params)
    theta_1p["seed"] = np.arange(len(theta_1p))  # varies within every group, must be ignored

    mapping, ambiguous = infer_1p_parameter_names(theta_1p)

    assert ambiguous == {}
    assert mapping == {1: "Omega0", 2: "sigma8"}


def test_infer_1p_parameter_names_ignores_non_numeric_columns():
    # Regression: a real params file's raw label column (e.g. "#Name",
    # values like "1P_p1_3") used to survive into theta_1p.columns and
    # crash max()-min() with a str-minus-str TypeError. params.py now
    # drops that column at load time, but this is the second line of
    # defense: any other non-numeric column must be silently excluded,
    # not crash the whole mapping.
    params = ["Omega0", "sigma8"]
    theta_1p = _synthetic_1p_theta(params)
    theta_1p["raw_label"] = [f"1P_{lab}" for lab in theta_1p.index]

    mapping, ambiguous = infer_1p_parameter_names(theta_1p)

    assert ambiguous == {}
    assert mapping == {1: "Omega0", 2: "sigma8"}


def test_infer_1p_parameter_names_flags_ambiguous_group():
    params = ["Omega0", "sigma8"]
    theta_1p = _synthetic_1p_theta(params, steps=(-1, 1))

    # Break group 1: make a second column vary too, so it's no longer
    # exactly-one-varying-column.
    group1 = [lab for lab in theta_1p.index if lab.startswith("p1_")]
    theta_1p.loc[group1, "sigma8"] = np.linspace(0.1, 0.9, len(group1))

    mapping, ambiguous = infer_1p_parameter_names(theta_1p)

    assert 1 in ambiguous
    assert set(ambiguous[1]) == {"Omega0", "sigma8"}
    assert 1 not in mapping
    assert mapping[2] == "sigma8"


def test_infer_1p_parameter_names_handles_double_digit_indices():
    # 10 parameters so index 10 exists alongside single-digit ones -- a
    # startswith("p1_") style prefix filter would wrongly also match "p10_*"
    params = [f"Astro{i}" for i in range(1, 11)]
    theta_1p = _synthetic_1p_theta(params)

    mapping, ambiguous = infer_1p_parameter_names(theta_1p)

    assert ambiguous == {}
    assert mapping[1] == "Astro1"
    assert mapping[10] == "Astro10"


def test_group_steps_filters_sorts_and_includes_shared_fiducial():
    sim_ids = ["p2_1", "p1_3", "p1_n1", "p3_0", "p1_1", "0"]
    assert group_steps(sim_ids, 1) == ["p1_n1", "0", "p1_1", "p1_3"]
    # "p3_0" and "0" are tied at step 0; the sort is stable but their
    # relative order isn't semantically meaningful, so compare as a set
    assert set(group_steps(sim_ids, 3)) == {"0", "p3_0"}
    assert group_steps(sim_ids, 9) == ["0"]  # fiducial still shared even for an unseen parameter


def test_group_steps_double_digit_index_not_confused_with_single_digit():
    sim_ids = ["p1_1", "p10_1", "p10_2", "0"]
    assert group_steps(sim_ids, 1) == ["0", "p1_1"]
    assert set(group_steps(sim_ids, 10)) == {"0", "p10_1", "p10_2"}


def test_mean_cdf_by_step_averages_repeated_steps_and_includes_fiducial():
    n_k, n_r = 2, 3
    # param 1 has two sims at step 1 (different seeds), one at step 2, and
    # the shared fiducial ("0") which must be pulled in as its step-0 point
    sim_ids = ["p1_1", "p1_1", "p1_2", "p2_1", "0"]
    summaries = np.array([
        np.full(n_k * n_r, 1.0),
        np.full(n_k * n_r, 3.0),    # step 1 mean should be 2.0
        np.full(n_k * n_r, 5.0),    # step 2
        np.full(n_k * n_r, 99.0),   # different parameter, must be excluded
        np.full(n_k * n_r, 7.0),    # shared fiducial -> step 0 for param 1 too
    ])

    steps, mean_cdfs = mean_cdf_by_step(summaries, sim_ids, param_index=1, n_k=n_k, n_r=n_r)

    assert list(steps) == [0, 1, 2]
    assert np.allclose(mean_cdfs[0], 7.0)   # fiducial
    assert np.allclose(mean_cdfs[1], 2.0)   # step 1, averaged over 2 seeds
    assert np.allclose(mean_cdfs[2], 5.0)   # step 2


def test_monotonic_trend_detects_a_real_trend():
    values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    response = values * 2 + 0.01  # monotone increasing, tiny noise-free offset

    rho, p = monotonic_trend(values, response)
    assert rho > 0.99
    assert p < 0.01


def test_monotonic_trend_requires_at_least_three_steps():
    with pytest.raises(ValueError):
        monotonic_trend(np.array([0.1, 0.2]), np.array([1.0, 2.0]))


def test_step_label_round_trips_with_parse_1p_label():
    assert step_label(1, 3) == "p1_3"
    assert step_label(1, -2) == "p1_n2"
    assert step_label(10, 1) == "p10_1"
    for pidx, step in [(1, 3), (1, -2), (10, 1), (28, -4)]:
        assert parse_1p_label(step_label(pidx, step)) == (pidx, step)


def test_step_label_rejects_zero():
    # step 0 is the shared fiducial and has no single reconstructable
    # label -- callers must use fiducial_value() instead
    with pytest.raises(ValueError):
        step_label(1, 0)


def test_fiducial_value_works_without_a_literal_zero_row():
    # Regression: the real parameter table for this 1P set has no row
    # literally labeled "0" -- it lists a separate fiducial repeat per
    # parameter instead (e.g. "p1_0", "p2_0", ...), even though the
    # *simulations* dedupe the fiducial to one shared snapshot. A lookup
    # by literal "0" label raised KeyError; fiducial_value() must not.
    fiducial = {"Omega0": 0.31, "sigma8": 0.8, "A": 1.5}
    rows = {}
    for pidx, target in enumerate(fiducial, start=1):
        for step in (-2, -1, 0, 1, 2):
            label = f"p{pidx}_0" if step == 0 else step_label(pidx, step)
            row = dict(fiducial)
            row[target] = fiducial[target] + step * 0.02
            rows[label] = row
    theta = pd.DataFrame(rows).T[list(fiducial)]

    assert "0" not in theta.index
    for name, expected in fiducial.items():
        assert fiducial_value(theta, name) == pytest.approx(expected)


def test_fiducial_value_works_with_a_literal_zero_row_too():
    # the other real convention (one shared "0" row) must also work
    fiducial = {"Omega0": 0.31, "sigma8": 0.8}
    rows = {"0": dict(fiducial)}
    for pidx, target in enumerate(fiducial, start=1):
        for step in (-1, 1):
            row = dict(fiducial)
            row[target] = fiducial[target] + step * 0.02
            rows[step_label(pidx, step)] = row
    theta = pd.DataFrame(rows).T[list(fiducial)]

    for name, expected in fiducial.items():
        assert fiducial_value(theta, name) == pytest.approx(expected)


def test_fiducial_value_raises_on_ambiguous_tie():
    # two equally-common values -- no clear fiducial, must not silently
    # guess
    theta = pd.DataFrame({"X": [1.0, 1.0, 2.0, 2.0]})
    with pytest.raises(ValueError):
        fiducial_value(theta, "X")
