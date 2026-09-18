"""
selection_bias.py
=================

Diagnose and correct for selection bias in activity-selected
SMBH samples.

Problem
-------
Activity selections (luminosity, fEdd) discard simulations
that fall below a minimum BH count.  If the probability of
retention correlates with a CAMELS parameter, the surviving
sample is biased and the quartile-based sensitivity analysis
confounds selection with clustering.

Solution
--------
1.  Diagnose:  Spearman ρ + KS tests of retained vs full
    parameter distributions.
2.  Correct:   Inverse-propensity weighting (IPW) via logistic
    regression.
3.  Verify:    Re-run sensitivity on the weighted sample and
    compare to unweighted results.

Usage
-----
    from src.selection_bias import (
        diagnose_selection_bias,
        compute_ipw_weights,
        weighted_sensitivity_table,
    )
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from sklearn.linear_model import LogisticRegression

from src.sensitivity_bootstrap import (
    _rms_response,
    sensitivity_table,
)


# ==============================================================
# Diagnosis
# ==============================================================

def diagnose_activity_cut_bias(sim_ids, nbh, theta_all, params=None):
    """Does N_BH surviving the activity cut correlate with theta?"""
    theta = theta_all.loc[sim_ids]
    rows = []
    for p in (params or theta.columns):
        rho, sp = spearmanr(nbh, theta[p].values)
        rows.append({"parameter": p, "spearman_rho": rho, "spearman_p": sp,
                      "bias_flag": sp < 0.01})
    return pd.DataFrame(rows)



def diagnose_selection_bias(
    retained_ids,
    all_ids,
    theta_all,
    params=None,
):
    """
    Test whether the retained simulation subset is a biased
    draw from the full LH parameter space.

    Parameters
    ----------
    retained_ids : array of sim_ids that survived the selection
    all_ids      : array of all 1000 (or however many) sim_ids
    theta_all    : DataFrame indexed by sim_id with all parameters
                   (for the full LH set, not just retained)
    params       : list of parameter names to test

    Returns
    -------
    diag_df : DataFrame with columns:
        parameter, spearman_rho, spearman_p,
        ks_stat, ks_p, retained_mean, full_mean, bias_flag
    """
    if params is None:
        params = list(theta_all.columns)

    retained_set = set(retained_ids)
    is_retained  = np.array([sid in retained_set for sid in all_ids])
    frac_retained = is_retained.mean()

    rows = []
    for p in params:
        full_vals     = theta_all.loc[all_ids, p].values
        retained_vals = theta_all.loc[retained_ids, p].values

        # Spearman: does retention correlate with parameter?
        # (undefined when retention is 100% — no variation in is_retained)
        if frac_retained > 0.999 or frac_retained < 0.001:
            rho, sp = 0.0, 1.0
        else:
            rho, sp = spearmanr(is_retained.astype(float), full_vals)

        # KS: are the distributions different?
        ks_stat, ks_p = ks_2samp(retained_vals, full_vals)

        rows.append({
            "parameter":    p,
            "spearman_rho": rho,
            "spearman_p":   sp,
            "ks_stat":      ks_stat,
            "ks_p":         ks_p,
            "retained_mean": np.mean(retained_vals),
            "full_mean":     np.mean(full_vals),
            "bias_flag":     abs(rho) > 0.1 or ks_p < 0.01,
        })

    diag_df = pd.DataFrame(rows)

    return diag_df


# ==============================================================
# Propensity score / IPW weights
# ==============================================================

def compute_ipw_weights(
    retained_ids,
    all_ids,
    theta_all,
    params=None,
    clip_min=0.05,
    clip_max=0.95,
):
    """
    Compute inverse-propensity weights for retained simulations.

    Fits a logistic regression:
        P(retained | θ) = σ(β·θ)

    then weights each retained sim by 1 / P̂(retained | θ_i).

    Weights are clipped to [1/clip_max, 1/clip_min] to avoid
    instability from extreme propensity scores.

    Parameters
    ----------
    retained_ids : sim_ids that survived selection
    all_ids      : all sim_ids in the LH set
    theta_all    : full parameter DataFrame
    params       : parameter columns to use
    clip_min/max : propensity score clipping bounds

    Returns
    -------
    weights : ndarray, shape (n_retained,), ordered by retained_ids
    propensity : ndarray, propensity scores for retained sims
    model : fitted LogisticRegression
    """
    if params is None:
        params = list(theta_all.columns)

    retained_set = set(retained_ids)
    is_retained  = np.array([sid in retained_set for sid in all_ids])

    frac_retained = is_retained.mean()

    # If retention is near-complete, IPW is unnecessary and
    # logistic regression would fail (single-class input).
    if frac_retained > 0.98:
        n = len(retained_ids)
        return (
            np.ones(n),               # uniform weights
            np.full(n, frac_retained), # propensity ≈ 1
            None,                      # no model fitted
        )

    X = theta_all.loc[all_ids, params].values

    # Standardise features for stable logistic regression
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mu) / std

    model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, solver="lbfgs",
    )
    model.fit(X_std, is_retained.astype(int))

    # Propensity scores for retained sims only
    retained_mask = np.isin(all_ids, retained_ids)
    X_ret = (theta_all.loc[retained_ids, params].values - mu) / std
    propensity = model.predict_proba(X_ret)[:, 1]

    # Clip to avoid extreme weights
    propensity = np.clip(propensity, clip_min, clip_max)

    weights = 1.0 / propensity

    # Normalise so weights sum to n_retained (preserves scale)
    weights *= len(weights) / weights.sum()

    return weights, propensity, model


# ==============================================================
# Weighted RMS response
# ==============================================================

def _weighted_rms_response(residuals, param_vals, weights):
    """
    Quartile-based RMS response with IPW weights.

    Uses unweighted quartile cuts (to maintain the same
    quartile definition as the unweighted analysis), but
    computes weighted means within quartiles.
    """
    q25 = np.percentile(param_vals, 25)
    q75 = np.percentile(param_vals, 75)

    low  = param_vals < q25
    high = param_vals > q75

    if low.sum() < 2 or high.sum() < 2:
        return np.nan

    # Weighted means
    w_high = weights[high] / weights[high].sum()
    w_low  = weights[low]  / weights[low].sum()

    mean_high = (residuals[high] * w_high[:, np.newaxis]).sum(axis=0)
    mean_low  = (residuals[low]  * w_low[:, np.newaxis]).sum(axis=0)

    diff = mean_high - mean_low
    return np.sqrt(np.mean(diff ** 2))


# ==============================================================
# Weighted sensitivity table with bootstrap
# ==============================================================

def weighted_sensitivity_table(
    residuals,
    theta,
    weights,
    params=None,
    n_boot=2000,
    n_null=2000,
    ci_level=0.95,
    boot_seed=42,
    null_seed=123,
):
    """
    Like sensitivity_table() but uses IPW-weighted quartile means.

    Bootstrap: resamples sims (rows), keeping their weights.
    Null:      shuffles parameter labels, keeping weights fixed.
    """
    if params is None:
        params = list(theta.columns)

    alpha = 1 - ci_level
    rng_boot = np.random.default_rng(boot_seed)
    rng_null = np.random.default_rng(null_seed)
    n = len(residuals)

    rows = []
    for p in params:
        pv = theta[p].values

        # Observed
        rms_obs = _weighted_rms_response(residuals, pv, weights)

        # Bootstrap
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng_boot.choice(n, size=n, replace=True)
            boot[i] = _weighted_rms_response(
                residuals[idx], pv[idx], weights[idx],
            )
        ci_lo = np.nanpercentile(boot, 100 * alpha / 2)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2))

        # Null (permute parameter labels, keep weights)
        null = np.empty(n_null)
        for i in range(n_null):
            perm = rng_null.permutation(pv)
            null[i] = _weighted_rms_response(
                residuals, perm, weights,
            )
        null_95 = np.nanpercentile(null, 95)
        p_val   = np.mean(null >= rms_obs)

        rows.append({
            "parameter":   p,
            "R_obs":       rms_obs,
            "ci_lo":       ci_lo,
            "ci_hi":       ci_hi,
            "null_floor":  null_95,
            "p_value":     p_val,
            "significant": rms_obs > null_95,
        })

    return pd.DataFrame(rows)


# ==============================================================
# Convenience: full pipeline
# ==============================================================

def corrected_sensitivity(
    residuals,
    theta,
    retained_ids,
    all_ids,
    theta_all,
    params=None,
    n_boot=2000,
    n_null=2000,
):
    """
    Run both unweighted and IPW-corrected sensitivity analyses
    and return them side by side.

    Returns
    -------
    unweighted_df, weighted_df, diag_df, weights
    """
    if params is None:
        params = list(theta.columns)

    # Diagnosis
    diag_df = diagnose_selection_bias(
        retained_ids, all_ids, theta_all, params,
    )

    # Unweighted
    uw_df = sensitivity_table(
        residuals, theta, params=params,
        n_boot=n_boot, n_null=n_null,
    )

    # IPW weights
    weights, propensity, model = compute_ipw_weights(
        retained_ids, all_ids, theta_all, params,
    )

    # Weighted
    w_df = weighted_sensitivity_table(
        residuals, theta, weights, params=params,
        n_boot=n_boot, n_null=n_null,
    )

    return uw_df, w_df, diag_df, weights


# ==============================================================
# nbh-size confound correction (for activity cuts with ~100%
# whole-sim retention, where compute_ipw_weights() is a no-op)
# ==============================================================

def residualize_theta_on_nbh(theta, nbh, params=None):
    """
    Regress each parameter in `theta` on `nbh` (the number of BHs
    that survived the activity cut for that sim) and return the
    residuals in place of the raw parameter values.

    Use this when `diagnose_activity_cut_bias` flags a parameter
    (e.g. Omega_m) as correlated with nbh but `diagnose_selection_bias`
    shows ~100% whole-sim retention -- in that regime
    `compute_ipw_weights` short-circuits to uniform weights (see its
    `frac_retained > 0.98` branch) and can't remove the confound,
    since there's no dropout to reweight. Residualizing theta instead
    strips out the part of each parameter's variation that's
    predictable from nbh, so any R(p) computed on theta_resid can't
    be driven by nbh varying systematically with theta.

    The output is a drop-in replacement for `theta`: pass it to the
    existing, unmodified `sensitivity_table()` from
    `sensitivity_bootstrap.py` exactly as you would the original.

    Parameters
    ----------
    theta  : DataFrame indexed by sim_id, columns = parameter names
    nbh    : array-like, same length/order as theta, BH counts
             surviving the activity cut for each sim
    params : list of parameter names to residualize
             (default: all columns of theta)

    Returns
    -------
    theta_resid : DataFrame, same shape/index as theta, with each
                  `params` column replaced by its residual after a
                  linear fit against nbh (other columns untouched)
    """
    if params is None:
        params = list(theta.columns)

    nbh = np.asarray(nbh, dtype=float)
    X = np.column_stack([np.ones_like(nbh), nbh])

    theta_resid = theta.copy()
    for p in params:
        y = theta[p].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        theta_resid[p] = y - X @ beta

    return theta_resid


def nbh_corrected_sensitivity(
    residuals,
    theta,
    nbh,
    params=None,
    n_boot=2000,
    n_null=2000,
):
    """
    Companion to `corrected_sensitivity()` for the nbh-size-bias
    case: runs `sensitivity_table()` (unmodified, from
    sensitivity_bootstrap.py) once on the raw theta and once on
    theta residualized against nbh, so the two can be compared
    directly.

    Note: `_rms_response`'s quartile split (25th/75th percentile)
    is computed on whatever `param_vals` it's given -- so on the
    corrected table, "high vs low Omega_m quartile" means high vs
    low *after* removing the nbh-predictable part. That's the point
    of the correction, but worth stating explicitly if this goes in
    the paper.

    Returns
    -------
    raw_df, corrected_df, theta_resid
    """
    if params is None:
        params = list(theta.columns)

    raw_df = sensitivity_table(
        residuals, theta, params=params,
        n_boot=n_boot, n_null=n_null,
    )

    theta_resid = residualize_theta_on_nbh(theta, nbh, params=params)

    corrected_df = sensitivity_table(
        residuals, theta_resid, params=params,
        n_boot=n_boot, n_null=n_null,
    )

    return raw_df, corrected_df, theta_resid