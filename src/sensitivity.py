"""
How does the (abundance-removed) kNN-CDF respond to each parameter, and
at what clustering scale r?

Core statistic: split simulations by a parameter's 25th/75th percentile,
take the high-quartile-mean minus low-quartile-mean of the residual
summary. Kept *per bin*, reshaped to (n_k, n_r), rather than immediately
collapsed to a single RMS number -- the radial axis is the one part of
this statistic with direct physical meaning (a scale), and collapsing it
throws that away. The scalar

    R(p) = sqrt(mean(diff ** 2))

is still computed (as `R_scalar` in `sensitivity_table`) since it's a
convenient single-number ranking of parameters, but it's derived from the
scale-resolved array, not computed separately.

Both a bootstrap CI (resample simulations with replacement) and a
permutation null (shuffle parameter labels) are computed per bin, so
"parameter p imprints at scale r" is backed by a real null comparison and
not just eyeballing a curve.
"""

import numpy as np
import pandas as pd


def _quartile_diff(residuals, param_vals):
    """Signed high-quartile-mean minus low-quartile-mean, per bin."""
    q25, q75 = np.percentile(param_vals, [25, 75])
    low = param_vals < q25
    high = param_vals > q75

    if low.sum() < 2 or high.sum() < 2:
        return np.full(residuals.shape[1], np.nan)

    return residuals[high].mean(axis=0) - residuals[low].mean(axis=0)


def infer_layout(kvals, rgrid):
    """(n_k, n_r) for the k-major concatenated summary layout used by knn_summary."""
    return len(kvals), len(rgrid)


def scale_resolved_response(residuals, param_vals, n_k):
    """Per-bin quartile diff, reshaped to (n_k, n_r)."""
    n_total = residuals.shape[1]
    if n_total % n_k != 0:
        raise ValueError(f"n_total_bins={n_total} not divisible by n_k={n_k}")
    n_r = n_total // n_k
    return _quartile_diff(residuals, param_vals).reshape(n_k, n_r)


def bootstrap_scale(residuals, param_vals, n_k, n_boot=2000, seed=42):
    """
    Bootstrap replicates of the scale-resolved response.

    Returns (obs, boot) where obs is (n_k, n_r) and boot is
    (n_boot, n_k, n_r).
    """
    rng = np.random.default_rng(seed)
    n = len(residuals)

    obs = scale_resolved_response(residuals, param_vals, n_k)
    boot = np.empty((n_boot, *obs.shape))
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = scale_resolved_response(residuals[idx], param_vals[idx], n_k)

    return obs, boot


def null_scale(residuals, param_vals, n_k, n_null=2000, seed=123):
    """Permutation-null replicates: shuffle parameter labels, keep residuals fixed."""
    rng = np.random.default_rng(seed)
    shape = scale_resolved_response(residuals, param_vals, n_k).shape

    null = np.empty((n_null, *shape))
    for i in range(n_null):
        perm = rng.permutation(param_vals)
        null[i] = scale_resolved_response(residuals, perm, n_k)

    return null


def sensitivity_table(
    residuals,
    theta,
    n_k,
    rgrid,
    kvals,
    params=None,
    n_boot=2000,
    n_null=2000,
    ci_level=0.95,
    boot_seed=42,
    null_seed=123,
):
    """
    Full scale-resolved + scalar sensitivity result for each parameter.

    Parameters
    ----------
    residuals : (n_sims, n_k * n_r) abundance-removed kNN-CDF residuals
    theta     : DataFrame indexed by sim_id, aligned to `residuals`'s row
                order (see `src.params.align_to_params`)
    n_k, rgrid, kvals : summary layout (from `infer_layout` / config)
    params    : parameter names to analyze (default: all of theta's columns)

    Returns
    -------
    dict: {"params", "rgrid", "kvals", "n_k", <param>: {...}} where each
    per-parameter entry has "obs", "ci_lo", "ci_hi", "null_floor",
    "p_value", "significant" (all (n_k, n_r)) and scalar "R_scalar",
    "R_ci_lo", "R_ci_hi", "R_p_value".
    """
    if params is None:
        params = list(theta.columns)

    alpha = 1 - ci_level
    table = {"params": list(params), "rgrid": np.asarray(rgrid), "kvals": list(kvals), "n_k": n_k}

    for p in params:
        pv = theta[p].values

        obs, boot = bootstrap_scale(residuals, pv, n_k, n_boot=n_boot, seed=boot_seed)
        null = null_scale(residuals, pv, n_k, n_null=n_null, seed=null_seed)

        ci_lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

        null_abs = np.abs(null)
        null_floor = np.nanpercentile(null_abs, 95, axis=0)
        with np.errstate(invalid="ignore"):
            p_value = np.nanmean(null_abs >= np.abs(obs)[None, :, :], axis=0)
        significant = np.abs(obs) > null_floor

        r_scalar = np.sqrt(np.nanmean(obs**2))
        r_boot = np.sqrt(np.nanmean(boot.reshape(boot.shape[0], -1) ** 2, axis=1))
        r_ci_lo, r_ci_hi = np.nanpercentile(r_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        r_null = np.sqrt(np.nanmean(null.reshape(null.shape[0], -1) ** 2, axis=1))
        r_p_value = np.mean(r_null >= r_scalar)
        r_null_floor = np.nanpercentile(r_null, 95)

        table[p] = {
            "obs": obs,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "null_floor": null_floor,
            "p_value": p_value,
            "significant": significant,
            "R_scalar": r_scalar,
            "R_ci_lo": r_ci_lo,
            "R_ci_hi": r_ci_hi,
            "R_p_value": r_p_value,
            "R_null_floor": r_null_floor,
        }

    return table


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Testing all 6 parameters against the same summaries means ~26% chance of
    at least one p < 0.05 under a global null, so raw p-values overstate
    significance. BH controls the false discovery rate instead, and is valid
    under the positive dependence these correlated tests have.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)

    scaled = p[order] * m / np.arange(1, m + 1)
    # q-values must be monotone in p: sweep the running minimum down from the largest
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]

    q = np.empty(m)
    q[order] = np.clip(scaled, 0, 1)
    return q


def summary_dataframe(table):
    """
    The scalar R(p) part of `sensitivity_table`'s output as a tidy DataFrame.

    `significant` uses the FDR-adjusted q-value, not the raw p-value. Note
    that `ci_lo`/`ci_hi` are bootstrap bounds on R itself: since R is an RMS
    (positive-definite), its CI excludes zero even under a pure null, so the
    CI says nothing about significance -- compare R against `null_floor`, or
    read `q_value`.
    """
    params = table["params"]
    p_values = [table[p]["R_p_value"] for p in params]
    q_values = benjamini_hochberg(p_values)

    rows = [
        {
            "parameter": p,
            "R_obs": table[p]["R_scalar"],
            "ci_lo": table[p]["R_ci_lo"],
            "ci_hi": table[p]["R_ci_hi"],
            "null_floor": table[p]["R_null_floor"],
            "p_value": table[p]["R_p_value"],
            "q_value": q,
            "significant": q < 0.05,
        }
        for p, q in zip(params, q_values)
    ]
    return pd.DataFrame(rows)
