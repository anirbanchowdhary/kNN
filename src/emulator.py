"""
Point prediction: given a measured kNN-CDF residual vector, what are the
predicted values of (Omega_m, sigma_8)?

This is a genuinely different question from `fisher.py`'s Fisher-matrix
forecast. Fisher answers "how tightly *could* this statistic constrain
the parameters" (an achievable-precision bound, from a local-linear
response and a diagonal noise covariance); this module answers "given
one actual measured summary, what values does that predict" -- a point
predictor, trained on the LH suite's (summary, theta) pairs, evaluated on
genuinely held-out data. Its held-out RMSE is an empirical scatter, not a
rigorous bound, and it is not a calibrated posterior: it returns a point
estimate per parameter, not a distribution.

Inputs are the kNN-CDF residual vector only (the same (n_sims, n_k*n_r)
arrays every other module here uses). The 4 astrophysics parameters
(A_SN1, A_AGN1, A_SN2, A_AGN2) are neither inputs nor targets -- the
model implicitly marginalizes over their variation across the LH suite's
Latin-hypercube design, the same "control for everything else" logic
`fisher.linear_response` makes explicit via OLS regression. This is only
valid for LH's own astrophysics-parameter distribution; a summary from
outside that prior range has no such guarantee.

Every function takes `model_factory` (default: `default_model`, a
`RandomForestRegressor`) rather than hardcoding an estimator, so a
different scikit-learn-API regressor drops in without touching the
cross-validation code. Random forests are the default here rather than a
Gaussian process because they need no kernel/length-scale tuning at
~150-300 input bins and n~950-1000 samples, where a default-kernel GP is
not competitive without feature reduction first, and they never split on
a constant feature, so the pinned-CDF-bin problem `fisher.py` needed
`drop_uninformative_bins` for (33/150 AGN bins, 23/150 galaxy bins were
exactly zero-variance in the real CV run) is a non-issue here "for free".

`cross_val_predict_emulator` fits one model *per target independently*,
not a single joint multi-output model, even though `RandomForestRegressor`
supports multi-output natively -- a joint fit's splitting criterion is
evaluated jointly across targets, so a much-higher-variance target can
dominate split choices and badly degrade a lower-variance one (verified
directly in `tests/test_emulator.py`). Independent per-target fits, all
sharing the same K-fold split, sidestep this at negligible extra cost.

Parallelism is deliberately split across two knobs, never both maxed at
once: `default_model`'s `RandomForestRegressor` defaults to `n_jobs=1`
(single-threaded per fit), while `cross_val_predict_emulator` (and
`null_control_metrics`, which calls it repeatedly) default to `n_jobs=-1`
themselves, parallelizing across K-fold *folds* rather than each fold's
trees. The other way round -- what this module originally shipped with,
and what notebook 08's first real run tried -- nests an implicitly
sequential fold loop inside a per-fit `n_jobs=-1`: correct on a small
machine where a single fit's own internal parallelism already saturates
every core, but on any machine with more cores than one fit can
profitably use internally (the realistic case for this project's actual
hardware), the fold loop burns only that one fit's ceiling while leaving
the rest of the machine idle for the whole run -- and it is, independent
of that, the documented sklearn/joblib nested-parallelism anti-pattern.
The one exception is `train_full_model`'s single one-off fit
(`feature_importance`, notebook 08 section 6) -- not inside a loop, so
full per-fit parallelism (`functools.partial(default_model, n_jobs=-1)`)
is safe and the right choice there.

The single most important check this module supports is
`null_control_metrics`: an emulator that looks predictive on shuffled
targets is a leakage or overfitting bug, not a good model -- the same
permutation-null discipline `sensitivity.py`/`onep.py` apply everywhere
else in this project. Never trust `prediction_metrics` computed from
anything but `cross_val_predict_emulator`'s held-out predictions --
`train_full_model`'s model has seen every LH simulation and its own
predictions on that same data say nothing about generalization.

Comparing several feature sets (notebook 08 section 3 -- AGN-only,
galaxy-only, combined) by calling `cross_val_predict_emulator` once per
feature set is still only fold-parallel *within* one feature set:
`n_splits` fits are in flight together, but the next feature set's fits
don't start until the current one's both targets finish, so any cores
beyond what `n_splits` can keep busy sit idle between feature sets. On
real hardware with more cores than one feature set's folds can use (or
just more feature sets than a naive one-at-a-time loop pipelines well),
that idle time is real wall-clock cost with nothing hiding it -- and
because `cross_val_predict`'s own progress output is scoped to a single
call, there's also no visibility into whether a feature set is actually
running or stuck. `cross_val_predict_many` flattens every (feature set,
target, fold) triple into ONE independent unit of work and dispatches
them all in a single `joblib.Parallel` call, so joblib load-balances the
whole section across every available core at once, and a single
`verbose=` setting reports progress across the entire section rather
than one feature set at a time. It computes exactly the same predictions
`cross_val_predict_emulator` would (same `KFold` splits, same per-fold
fit-and-predict) -- purely a re-dispatch of the same work, not a
different model.

`n_jobs=-1` tells joblib to target `os.cpu_count()` workers, which is
the HOST's logical CPU count and can badly overstate what's actually
usable inside a container (e.g. a real run reported 256 detected cores
while a single 500-tree fit still took over a minute) -- spawning far
more worker processes than there are independent units of work is pure
overhead with no upside. `effective_cpu_count()` checks cgroup CPU
quotas and CPU affinity, not just `os.cpu_count()`, for a more realistic
ceiling; callers choosing an explicit `n_jobs` (rather than trusting
joblib's own `-1` handling) should prefer
`min(effective_cpu_count(), <number of independent tasks>)` over a bare
`os.cpu_count()`.
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from .complementarity import align_common_sims


def effective_cpu_count():
    """
    A more realistic worker count than `os.cpu_count()` for deciding how
    many processes to actually spawn -- found the hard way on a real run
    that reported `os.cpu_count() == 256` while a single 500-tree fit
    still took over a minute: `os.cpu_count()` reports the HOST's total
    logical CPUs, which can badly overstate what this process can
    actually use at once inside a container. Checks, in order, and
    returns the SMALLEST positive count found (any one of these can be
    the actual ceiling):

    1. cgroup v2 CPU quota (`/sys/fs/cgroup/cpu.max`, "<quota> <period>"
       microseconds, or "max" for unlimited)
    2. cgroup v1 CPU quota (`cpu.cfs_quota_us` / `cpu.cfs_period_us`
       under `/sys/fs/cgroup/cpu/`, quota of -1 meaning unlimited)
    3. `os.sched_getaffinity(0)` (Linux only) -- the process's actual CPU
       affinity mask, which container CPU pinning restricts even when a
       quota doesn't
    4. `os.cpu_count()` as the final fallback (always available)

    This does not know how many of those cores are already busy with
    something else, only how many this process could ever use -- so it's
    a ceiling on `n_jobs`, not a live load measurement. Never returns
    less than 1.
    """
    candidates = []

    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            candidates.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            candidates.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass

    candidates.append(os.cpu_count() or 1)

    return min(candidates)


def default_model(random_state=42, n_estimators=500, n_jobs=1):
    """
    The default regressor: a random forest (see module docstring for why).

    `n_jobs=1` by default -- deliberately NOT `-1`. This factory is
    called once per K-fold *fold* inside `cross_val_predict_emulator`,
    and that function parallelizes across folds itself (its own
    `n_jobs`, default `-1`, passed straight through to
    `cross_val_predict`). If each individual fold's forest also grabbed
    every core, the two parallelism layers would nest -- the standard
    sklearn/joblib oversubscription anti-pattern (see module docstring).

    CHANGED: this default was `n_jobs=-1` before this project's first
    real notebook run made the cost of that visible (60 sequential
    500-tree fits, one per fold x target x feature-set, never completed
    in a tractable time). Anything constructing `default_model()`
    directly and expecting per-fit multithreading now needs
    `n_jobs=-1` explicit -- e.g. via `functools.partial(default_model,
    n_jobs=-1)` for a genuine one-off fit outside any fold loop
    (`train_full_model`).
    """
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)


def combined_features(sim_ids_agn, residuals_agn, sim_ids_gal, residuals_gal):
    """
    AGN+galaxy residuals, restricted to the simulations common to both
    fixed-N runs (they can retain different simulations), concatenated
    bin-wise into one feature matrix -- AGN's columns first, then
    galaxy's. A thin wrapper around `complementarity.align_common_sims`;
    never reimplement that intersection/reorder logic.

    Unlike `fisher.combine_fisher` (which sums two tracers' Fisher
    matrices and is only valid if they're independent -- real data
    already showed they aren't, median cross-tracer correlation 0.69),
    concatenating features lets a tree-based model learn cross-tracer
    structure directly. It does not inherit that independence assumption.

    Returns
    -------
    (common_sim_ids, X_combined)
    """
    common, aligned_agn, aligned_gal = align_common_sims(
        sim_ids_agn, residuals_agn, sim_ids_gal, residuals_gal
    )
    return common, np.hstack([aligned_agn, aligned_gal])


def cross_val_predict_emulator(X, y, model_factory=default_model, n_splits=10,
                                random_state=42, n_jobs=-1):
    """
    K-fold out-of-fold predictions: every row of `X`/`y` is predicted by
    a model that never saw it during training. Preferred over a single
    train/test split at this project's sample size (~950-1000 LH sims) --
    a single 80/20 split holds out only ~190-200 rows and its score is
    sensitive to which rows land in the test set; K-fold OOF uses the
    whole suite for both training and evaluation while still giving every
    row an honest held-out prediction.

    Uses an explicit `KFold(shuffle=True, random_state=...)`, never the
    bare integer `cv=n_splits` shortcut, which does not shuffle by
    default and would correlate folds with `sim_id` order.

    For a multi-target `y`, fits one model *per target independently*
    (looping over columns), all sharing the same K-fold split -- not a
    single joint multi-output model. sklearn's multi-output splitting
    criterion is evaluated jointly across targets, so a much-higher-
    variance target can dominate split choices and badly degrade a
    lower-variance one (verified in `tests/test_emulator.py`: a 1000x
    target-scale mismatch drops the low-variance target's R^2 from ~0.99
    to negative under a joint fit). Independent per-target fits sidestep
    this entirely, at negligible extra cost for the handful of targets
    this project ever predicts.

    `n_jobs` parallelizes across the `n_splits` folds (passed straight
    through to `cross_val_predict`), NOT across each fold's trees --
    `model_factory`'s default (`default_model`) builds its
    `RandomForestRegressor` with `n_jobs=1` for exactly this reason (see
    module docstring). Fold-level parallelism is the more robust place
    for it regardless of core count: the `n_splits` folds are fully
    independent, equal-cost units of work, so this scales with however
    many cores are available, whereas a single fit's own internal
    tree-parallelism has a lower ceiling that a large core count can
    outrun -- a purely sequential fold loop (as this module originally
    had) leaves any cores beyond that ceiling idle for the entire run.

    Parameters
    ----------
    X : (n_sims, n_features)
    y : (n_sims,) or (n_sims, n_targets)
    model_factory : callable(random_state) -> unfitted sklearn-API regressor
    n_jobs : int, default -1
        Passed to `cross_val_predict` -- parallelizes across the
        `n_splits` folds (one `cross_val_predict` call per target
        column). Set to `1` to force fully sequential fold fitting.

    Returns
    -------
    (n_sims,) or (n_sims, n_targets) ndarray, same shape and row order as `y`
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} -- not aligned")
    if n_splits > len(X):
        raise ValueError(f"n_splits={n_splits} exceeds the number of rows ({len(X)})")

    is_1d = y.ndim == 1
    y_2d = y.reshape(-1, 1) if is_1d else y

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.column_stack([
        cross_val_predict(model_factory(random_state=random_state), X, y_2d[:, j], cv=cv, n_jobs=n_jobs)
        for j in range(y_2d.shape[1])
    ])
    return preds[:, 0] if is_1d else preds


def _fit_predict_fold(model_factory, random_state, X, y_col, train_idx, test_idx):
    """
    One independent (feature set, target, fold) unit of work for
    `cross_val_predict_many` -- a module-level function, not a closure,
    so joblib's process-based backend can pickle and dispatch it.
    """
    model = model_factory(random_state=random_state)
    model.fit(X[train_idx], y_col[train_idx])
    return model.predict(X[test_idx])


def cross_val_predict_many(feature_sets, targets, model_factory=default_model,
                            n_splits=10, random_state=42, n_jobs=-1, verbose=0):
    """
    Out-of-fold predictions for several feature sets at once, dispatched
    as ONE flat, independent batch of (feature set, target, fold) fits
    rather than one `cross_val_predict_emulator` call per feature set
    (see module docstring for why that matters on real hardware).

    Uses the same `KFold(shuffle=True, random_state=...)` split and the
    same per-target-independent fitting `cross_val_predict_emulator`
    uses, so results are numerically identical to calling
    `cross_val_predict_emulator` once per feature set with the same
    `model_factory`/`n_splits`/`random_state` -- this is a re-dispatch of
    the same computation, not a different one.

    Parameters
    ----------
    feature_sets : dict {name: (X, theta)} -- theta must be a DataFrame
                   containing `targets` as columns, aligned row-for-row
                   with X (e.g. the `feature_sets` dict notebook 08
                   builds directly)
    targets       : list of target column names to predict, e.g.
                    ["Omega_m", "sigma_8"]
    verbose       : forwarded to `joblib.Parallel` -- e.g. `verbose=10`
                    prints a running "Done k out of N | elapsed ...
                    remaining ..." line as fits complete, covering every
                    feature set at once (the real per-fit progress signal
                    that a `tqdm` wrapped around this function's own
                    outer loop cannot give, since that loop no longer
                    exists -- all fits are dispatched together)

    Returns
    -------
    dict {name: (X, y, y_pred)}, same per-feature-set contents
    `cross_val_predict_emulator` produces (X as given, y as
    `theta[targets].values`, y_pred the same shape) -- so
    `prediction_metrics` and everything downstream is unchanged.
    """
    jobs = []
    job_meta = []
    outputs = {}

    for name, (X, theta) in feature_sets.items():
        X = np.asarray(X)
        y = theta[list(targets)].values
        if n_splits > len(X):
            raise ValueError(
                f"n_splits={n_splits} exceeds the number of rows ({len(X)}) for feature set '{name}'"
            )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X))
        outputs[name] = (X, y, np.empty_like(y, dtype=float))
        for t_idx in range(y.shape[1]):
            for train_idx, test_idx in splits:
                jobs.append(delayed(_fit_predict_fold)(
                    model_factory, random_state, X, y[:, t_idx], train_idx, test_idx
                ))
                job_meta.append((name, t_idx, test_idx))

    print(
        f"cross_val_predict_many: dispatching {len(jobs)} independent fold fits "
        f"({len(feature_sets)} feature sets x {len(targets)} targets x {n_splits} folds)..."
    )

    fold_preds = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)

    for (name, t_idx, test_idx), pred in zip(job_meta, fold_preds):
        outputs[name][2][test_idx, t_idx] = pred

    return outputs


def train_full_model(X, y, model_factory=default_model, random_state=42):
    """
    Fit one model on ALL of `X`/`y`. For downstream use only
    (`feature_importance`, or predicting on a genuinely separate,
    never-trained-on set) -- never for a performance metric. This model
    has seen every row, so its own predictions on `X` say nothing about
    generalization; use `cross_val_predict_emulator` for that.

    Unlike `cross_val_predict_emulator`, this fits ONE joint multi-output
    model when `y` has multiple columns (not independent per-target
    models) -- acceptable here since this function is explicitly not
    used to measure predictive accuracy, only to inspect
    `feature_importance`, and this project's two targets (Omega_m,
    sigma_8) have comparable prior scales, so the multi-output
    splitting-criterion pitfall `cross_val_predict_emulator`'s docstring
    describes is not expected to bias the importances meaningfully.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} -- not aligned")

    model = model_factory(random_state=random_state)
    model.fit(X, y)
    return model


def prediction_metrics(y_true, y_pred, target_names):
    """
    Per-target R^2 and RMSE, as a tidy DataFrame (columns: target, r2, rmse).

    Raises ValueError on a shape mismatch, or if any `y_true` column has
    zero variance (R^2 is undefined there -- never silently return a
    number that looks meaningful but isn't, the same discipline
    `fisher.marginalized_covariance` applies to a singular Fisher matrix).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.shape[1] != len(target_names):
        raise ValueError(
            f"y_true has {y_true.shape[1]} columns but {len(target_names)} target_names given"
        )

    rows = []
    for j, name in enumerate(target_names):
        if np.var(y_true[:, j]) == 0:
            raise ValueError(f"y_true column '{name}' has zero variance -- R^2 is undefined")
        rows.append({
            "target": name,
            "r2": r2_score(y_true[:, j], y_pred[:, j]),
            "rmse": np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j])),
        })
    return pd.DataFrame(rows)


def null_control_metrics(X, y, target_names, model_factory=default_model,
                          n_shuffles=100, n_splits=10, random_state=42, n_jobs=-1):
    """
    A permutation-null distribution of `prediction_metrics`: shuffle `y`'s
    *rows* (breaks the X<->y correspondence while keeping each column's
    own marginal distribution -- the same convention `sensitivity.null_scale`
    uses, shuffling parameter labels while keeping residuals fixed), rerun
    the identical `cross_val_predict_emulator` + `prediction_metrics`
    pipeline `n_shuffles` times.

    This is the central check of this module: a real emulator's observed
    R^2 must sit far above this null distribution's 95th percentile (and
    observed RMSE far below its 5th percentile), or the apparent
    "signal" is a pipeline artifact -- leakage, an overfit model, or
    chance -- not real predictive power.

    `n_jobs` is forwarded to every inner `cross_val_predict_emulator`
    call (see its docstring for the fold-vs-tree parallelism rationale)
    -- it matters more here than there: this function calls it
    `n_shuffles` times, so the fully-sequential-fold-loop version of the
    cost `cross_val_predict_emulator`'s docstring describes is multiplied
    by `n_shuffles` again.

    Returns
    -------
    tidy DataFrame with columns (target, r2, rmse, shuffle)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    frames = []
    for i in range(n_shuffles):
        perm = rng.permutation(len(y))
        y_shuffled = y[perm]
        y_pred = cross_val_predict_emulator(
            X, y_shuffled, model_factory=model_factory, n_splits=n_splits,
            random_state=random_state, n_jobs=n_jobs,
        )
        metrics = prediction_metrics(y_shuffled, y_pred, target_names)
        metrics["shuffle"] = i
        frames.append(metrics)

    return pd.concat(frames, ignore_index=True)


def feature_importance(model, n_k, n_r):
    """
    `model.feature_importances_` reshaped to (n_k, n_r) -- directly
    comparable, bin for bin, to `sensitivity.scale_resolved_response`'s
    layout. Raises if `model` has no `feature_importances_` (i.e. isn't a
    tree-based model).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"{type(model).__name__} has no feature_importances_ "
            f"(not a tree-based model) -- feature_importance only applies to those"
        )
    importances = model.feature_importances_
    if importances.shape[0] != n_k * n_r:
        raise ValueError(
            f"model has {importances.shape[0]} features, expected n_k*n_r={n_k * n_r}"
        )
    return importances.reshape(n_k, n_r)
