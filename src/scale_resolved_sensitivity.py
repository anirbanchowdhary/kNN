"""
scale_resolved_sensitivity.py
==============================

Scale-resolved companion to `sensitivity_bootstrap.py`.

Motivation
----------
`_rms_response` in `sensitivity_bootstrap.py` computes

    R(p) = sqrt( mean_bins( [<resid>_high_quartile - <resid>_low_quartile]^2 ) )

i.e. it collapses the per-bin high-minus-low quartile difference into a
single scalar by RMS-ing over *every* summary bin.  That throws away the
one axis with direct physical meaning -- the separation scale r.

This module reproduces the *exact same* quartile contrast but returns the
difference **per bin**, reshaped to (n_k, n_r), so you can see R as a
function of r for each kNN value k (or the single xi(r) curve for 2PCF).
The scalar R(p) is recovered exactly as

    R_scalar = sqrt( mean( diff.flatten()**2 ) )

so this is a strict decomposition of the existing metric, not a new one.

Layout convention
-----------------
The kNN summaries produced by `generate_knn*.run_suite` are
`np.concatenate([cdf(k_1), cdf(k_2), ...])`, each block of length
`len(rgrid)`.  So the flattened bin index is

    flat = k_index * n_r + r_index          (k-major, C order)

Pass `n_k = len(kvals)` for kNN summaries and `n_k = 1` for a flat
xi(r) 2PCF summary.  `n_r` is inferred as `n_total_bins // n_k`.

Nothing here is imported by the existing pipeline; it's purely additive.
The scalar `_rms_response` / `sensitivity_table` are left untouched.

Usage
-----
    from scale_resolved_sensitivity import (
        scale_resolved_response,     # single (p) -> (n_k, n_r) signed diff
        scale_sensitivity_table,     # all params -> dict of arrays + CI + null
        plot_scale_response,         # one parameter
        plot_scale_grid,             # all parameters
        infer_layout,                # (n_k, rgrid) from a loaded .npz
    )

    # res = load_and_analyze(path, ..., standardize_flag=True)   # your pipeline
    # n_k, rgrid = infer_layout(np.load(path, allow_pickle=True))
    # tab = scale_sensitivity_table(res["residuals"], res["theta"],
    #                               n_k=n_k, rgrid=rgrid,
    #                               params=PARAMS)
    # plot_scale_grid(tab, params=PARAMS)
"""

import numpy as np
import pandas as pd


# ==============================================================
# Layout helper
# ==============================================================

def infer_layout(npz_data):
    """
    Work out (n_k, rgrid) for a loaded summary .npz.

    - kNN files store 'kvals' -> n_k = len(kvals)
    - 2PCF files have no 'kvals' -> n_k = 1 (flat xi(r))

    `rgrid` is the per-k radial grid (bin centres), taken from the
    'rgrid' field that every generator saves.

    Returns
    -------
    n_k   : int
    rgrid : ndarray, shape (n_r,)
    """
    keys = set(npz_data.files) if hasattr(npz_data, "files") else set(npz_data)
    rgrid = np.asarray(npz_data["rgrid"])
    if "kvals" in keys:
        n_k = int(len(np.atleast_1d(npz_data["kvals"])))
    else:
        n_k = 1
    return n_k, rgrid


def _check_layout(n_total, n_k):
    if n_k < 1:
        raise ValueError(f"n_k must be >= 1, got {n_k}")
    if n_total % n_k != 0:
        raise ValueError(
            f"n_total_bins={n_total} not divisible by n_k={n_k}; "
            f"check that n_k matches the summary layout."
        )
    return n_total // n_k


# ==============================================================
# Core: per-bin quartile difference (the un-collapsed R metric)
# ==============================================================

def scale_resolved_response(residuals, param_vals, n_k=1):
    """
    Signed high-minus-low quartile mean difference, per bin.

    Identical quartile definition to `_rms_response`
    (25th/75th percentile split on `param_vals`), but the per-bin
    difference is returned reshaped to (n_k, n_r) instead of being
    RMS-collapsed to a scalar.

    Parameters
    ----------
    residuals  : ndarray (n_sims, n_total_bins)
    param_vals : ndarray (n_sims,)
    n_k        : int, number of k-blocks concatenated in the summary
                 (len(kvals) for kNN, 1 for 2PCF)

    Returns
    -------
    diff : ndarray (n_k, n_r)  signed  <resid>_high - <resid>_low  per bin
           (all-NaN of that shape if a quartile has < 2 sims)

    Notes
    -----
    The scalar R(p) of the existing pipeline is exactly
        np.sqrt(np.nanmean(diff**2))
    """
    n_total = residuals.shape[1]
    n_r = _check_layout(n_total, n_k)

    q25 = np.percentile(param_vals, 25)
    q75 = np.percentile(param_vals, 75)

    low  = param_vals < q25
    high = param_vals > q75

    if low.sum() < 2 or high.sum() < 2:
        return np.full((n_k, n_r), np.nan)

    diff = residuals[high].mean(axis=0) - residuals[low].mean(axis=0)
    return diff.reshape(n_k, n_r)


def scalar_from_scale(diff):
    """
    Collapse a scale-resolved diff back to the scalar R(p), reproducing
    `_rms_response` exactly (RMS over all bins).  Handy as a consistency
    check: this should equal sensitivity_bootstrap._rms_response(...).
    """
    return np.sqrt(np.nanmean(diff ** 2))


# ==============================================================
# Bootstrap + permutation null, per bin
# ==============================================================

def bootstrap_scale(residuals, param_vals, n_k=1, n_boot=2000, seed=42):
    """
    Non-parametric bootstrap of the per-bin quartile difference.

    Resamples *simulations* (rows) with replacement -- same scheme as
    `bootstrap_rms` -- preserving the cross-bin correlation structure.

    Returns
    -------
    obs   : ndarray (n_k, n_r)              observed signed diff
    boot  : ndarray (n_boot, n_k, n_r)      bootstrap replicates
    """
    rng = np.random.default_rng(seed)
    n = len(residuals)

    obs = scale_resolved_response(residuals, param_vals, n_k=n_k)

    boot = np.empty((n_boot, *obs.shape))
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = scale_resolved_response(
            residuals[idx], param_vals[idx], n_k=n_k
        )
    return obs, boot


def null_scale(residuals, param_vals, n_k=1, n_null=2000, seed=123):
    """
    Permutation null for the per-bin quartile difference.

    Shuffles parameter labels to break the parameter <-> residual
    association, exactly like `null_rms`, but keeps every bin separate.

    Returns
    -------
    null      : ndarray (n_null, n_k, n_r)  signed diff under the null
    """
    rng = np.random.default_rng(seed)

    shape = scale_resolved_response(residuals, param_vals, n_k=n_k).shape
    null = np.empty((n_null, *shape))
    for i in range(n_null):
        perm = rng.permutation(param_vals)
        null[i] = scale_resolved_response(residuals, perm, n_k=n_k)
    return null


# ==============================================================
# Table builder: all parameters
# ==============================================================

def scale_sensitivity_table(
    residuals,
    theta,
    n_k=1,
    rgrid=None,
    kvals=None,
    params=None,
    n_boot=2000,
    n_null=2000,
    ci_level=0.95,
    boot_seed=42,
    null_seed=123,
):
    """
    Scale-resolved analogue of `sensitivity_table`.

    For each parameter, returns the signed per-bin response together with
    a bootstrap CI band and a per-bin permutation null floor, all shaped
    (n_k, n_r).  Significance is judged *per bin* against that bin's own
    null floor, so you can see at which scales (and which k) a parameter's
    imprint exceeds chance.

    Parameters
    ----------
    residuals : ndarray (n_sims, n_total_bins)
    theta     : DataFrame indexed by sim_id, columns = parameter names
                (positionally aligned to `residuals`, as elsewhere in the
                 pipeline)
    n_k       : int, summary layout (len(kvals) for kNN, 1 for 2PCF)
    rgrid     : ndarray (n_r,), radial bin centres for labelling/plotting
    kvals     : sequence, the k values (for labelling only)
    params    : list of parameter names (default: all theta columns)
    ci_level  : bootstrap CI level (default 0.95)

    Returns
    -------
    table : dict with
        "params" : list of parameter names
        "rgrid"  : ndarray (n_r,)      (np.arange(n_r) if none supplied)
        "kvals"  : list                (list(range(n_k)) if none supplied)
        "n_k"    : int
        and, per parameter p, table[p] = dict of arrays each (n_k, n_r):
            "obs"        signed high-low diff
            "ci_lo"      bootstrap lower band
            "ci_hi"      bootstrap upper band
            "null_floor" 95th percentile of |null diff|  (>=0)
            "p_value"    per-bin permutation p-value on |diff|
            "significant" bool, |obs| > null_floor
            "R_scalar"   float, sqrt(mean(obs**2))  -- matches _rms_response
    """
    if params is None:
        params = list(theta.columns)

    n_total = residuals.shape[1]
    n_r = _check_layout(n_total, n_k)

    if rgrid is None:
        rgrid = np.arange(n_r)
    else:
        rgrid = np.asarray(rgrid)
        if len(rgrid) != n_r:
            raise ValueError(
                f"rgrid has length {len(rgrid)} but n_r={n_r} "
                f"(n_total={n_total}, n_k={n_k})."
            )
    if kvals is None:
        kvals = list(range(n_k))
    else:
        kvals = list(kvals)
        if len(kvals) != n_k:
            raise ValueError(
                f"kvals has length {len(kvals)} but n_k={n_k}."
            )

    alpha = 1 - ci_level
    table = {
        "params": list(params),
        "rgrid": rgrid,
        "kvals": kvals,
        "n_k": n_k,
    }

    for p in params:
        pv = theta[p].values

        obs, boot = bootstrap_scale(
            residuals, pv, n_k=n_k, n_boot=n_boot, seed=boot_seed,
        )
        null = null_scale(
            residuals, pv, n_k=n_k, n_null=n_null, seed=null_seed,
        )

        ci_lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

        # Per-bin null on the *magnitude* of the response.
        null_abs    = np.abs(null)
        null_floor  = np.nanpercentile(null_abs, 95, axis=0)

        # Per-bin permutation p-value: P(|null| >= |obs|)
        with np.errstate(invalid="ignore"):
            p_value = np.nanmean(null_abs >= np.abs(obs)[None, :, :], axis=0)

        significant = np.abs(obs) > null_floor

        table[p] = {
            "obs":         obs,
            "ci_lo":       ci_lo,
            "ci_hi":       ci_hi,
            "null_floor":  null_floor,
            "p_value":     p_value,
            "significant": significant,
            "R_scalar":    scalar_from_scale(obs),
        }

    return table


def to_long_dataframe(table):
    """
    Flatten a `scale_sensitivity_table` result into a tidy long-form
    DataFrame with one row per (parameter, k, r):

        parameter | k | r | obs | ci_lo | ci_hi | null_floor
                  | p_value | significant

    Convenient for CSV export / seaborn / cross-checking against the
    scalar table (group by parameter, then sqrt(mean(obs**2)) == R_obs).
    """
    rows = []
    rgrid = table["rgrid"]
    kvals = table["kvals"]
    for p in table["params"]:
        e = table[p]
        n_k, n_r = e["obs"].shape
        for ki in range(n_k):
            for ri in range(n_r):
                rows.append({
                    "parameter":   p,
                    "k":           kvals[ki],
                    "r":           rgrid[ri],
                    "obs":         e["obs"][ki, ri],
                    "ci_lo":       e["ci_lo"][ki, ri],
                    "ci_hi":       e["ci_hi"][ki, ri],
                    "null_floor":  e["null_floor"][ki, ri],
                    "p_value":     e["p_value"][ki, ri],
                    "significant": bool(e["significant"][ki, ri]),
                })
    return pd.DataFrame(rows)


# ==============================================================
# Plotting
# ==============================================================

def plot_scale_response(
    table,
    parameter,
    ax=None,
    figsize=(7, 4),
    logx=True,
    show_null=True,
    show_ci=True,
    cmap_name="viridis",
):
    """
    Plot the signed scale-resolved response for one parameter:
    one line per k, with a shaded 95% bootstrap band and the symmetric
    +/- per-bin null floor.

    The physically-readable content: where (in r, and for which k) the
    curve leaves the grey null band is where that parameter imprints on
    the clustering.  Sign tells you the direction (high-quartile raises
    or lowers the statistic at that scale).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    e     = table[parameter]
    rgrid = table["rgrid"]
    kvals = table["kvals"]
    n_k   = table["n_k"]

    cmap = plt.get_cmap(cmap_name)

    for ki in range(n_k):
        colour = cmap(ki / max(n_k - 1, 1))
        obs = e["obs"][ki]
        ax.plot(rgrid, obs, color=colour, lw=1.6,
                label=f"k={kvals[ki]}", zorder=3)
        if show_ci:
            ax.fill_between(rgrid, e["ci_lo"][ki], e["ci_hi"][ki],
                            color=colour, alpha=0.18, lw=0, zorder=2)

    if show_null:
        # Symmetric null band: use the median across k of the per-bin
        # null floor so the reference is a single grey envelope.
        nf = np.nanmedian(e["null_floor"], axis=0)
        ax.fill_between(rgrid, -nf, nf, color="0.6", alpha=0.25, lw=0,
                        zorder=1, label="95% null band")

    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\langle x\rangle_{\rm high} - \langle x\rangle_{\rm low}$")
    ax.set_title(f"Scale-resolved response: {parameter}"
                 f"  ($R={e['R_scalar']:.4f}$)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_scale_grid(
    table,
    params=None,
    ncols=3,
    figsize=None,
    logx=True,
    share_y=False,
):
    """
    Small-multiples grid: one `plot_scale_response` panel per parameter.
    Good as a single paper figure showing which parameters live on which
    scales.
    """
    import matplotlib.pyplot as plt

    if params is None:
        params = table["params"]

    n = len(params)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False, sharey=share_y)
    axes_flat = axes.ravel()

    for j, p in enumerate(params):
        plot_scale_response(table, p, ax=axes_flat[j], logx=logx)
        axes_flat[j].set_title(p, fontsize=10)
        # de-clutter: legend only on the first panel
        if j != 0 and axes_flat[j].get_legend() is not None:
            axes_flat[j].get_legend().remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig, axes


# ==============================================================
# Self-test on synthetic data
# ==============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_sims = 400
    n_k, n_r = 3, 50
    n_total = n_k * n_r
    rgrid = np.logspace(-1.5, 1.2, n_r)

    theta = pd.DataFrame({
        "Omega_m": rng.uniform(0.1, 0.5, n_sims),
        "sigma_8": rng.uniform(0.6, 1.0, n_sims),
        "A_SN1":   rng.uniform(0.25, 4.0, n_sims),
    })

    # Inject an Omega_m signal that lives ONLY in the large-r half of the
    # k=1 block, so the scale-resolved plot should light up there and the
    # rest should sit inside the null band.
    resid = rng.normal(0, 0.01, (n_sims, n_total))
    om_c  = (theta["Omega_m"] - theta["Omega_m"].mean()).values
    bump  = np.zeros(n_total)
    bump[(0 * n_r + n_r // 2):(0 * n_r + n_r)] = 1.0   # k=1, large r
    resid += 0.03 * np.outer(om_c, bump)

    print("Building scale-resolved table (synthetic)...")
    tab = scale_sensitivity_table(
        resid, theta, n_k=n_k, rgrid=rgrid, kvals=(1, 2, 4),
        params=["Omega_m", "sigma_8", "A_SN1"],
        n_boot=500, n_null=500,
    )

    # Consistency with the scalar metric: R_scalar must equal the RMS of
    # the observed per-bin diff, and match a hand-rolled _rms_response.
    for p in tab["params"]:
        obs = tab[p]["obs"]
        r_from_scale = np.sqrt(np.nanmean(obs ** 2))
        # hand-rolled scalar (same as sensitivity_bootstrap._rms_response)
        pv = theta[p].values
        q25, q75 = np.percentile(pv, [25, 75])
        d = resid[pv > q75].mean(0) - resid[pv < q25].mean(0)
        r_scalar = np.sqrt(np.mean(d ** 2))
        ok = np.isclose(r_from_scale, r_scalar) and \
             np.isclose(tab[p]["R_scalar"], r_scalar)
        nsig = int(tab[p]["significant"].sum())
        print(f"  {p:8s}  R={r_scalar:.5f}  "
              f"scale-consistent={ok}  n_bins_significant={nsig}")

    # Where did Omega_m light up? Expect: k=1 (row 0), large-r bins only.
    om_sig = tab["Omega_m"]["significant"]
    k1_hits = np.where(om_sig[0])[0]
    print(f"\nOmega_m significant bins in k=1 block at r-indices: "
          f"{k1_hits.min() if len(k1_hits) else '-'}.."
          f"{k1_hits.max() if len(k1_hits) else '-'} "
          f"(injected into {n_r//2}..{n_r-1})")

    print("\nLong-form head:")
    print(to_long_dataframe(tab).head(6).to_string(index=False))
    print("\nOK")