"""
sensitivity_bootstrap.py
========================

Bootstrap confidence intervals and permutation-based null
tests for the RMS parameter-sensitivity metric R(p).

Drop into src/ alongside parameter_sensitivity.py.

Usage
-----
    from src.sensitivity_bootstrap import (
        sensitivity_table,
        plot_sensitivity
    )

    table, fig = sensitivity_table(
        residuals, theta,
        params=["Omega_m","sigma_8","A_SN1","A_AGN1","A_SN2","A_AGN2"],
        n_boot=2000,
        n_null=2000,
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ==============================================================
# Core metric (self-contained so this module has no local deps)
# ==============================================================

def _rms_response(residuals, param_vals):
    """
    RMS of the mean-residual difference between upper
    and lower quartiles of a 1-d parameter array.
    """
    q25 = np.percentile(param_vals, 25)
    q75 = np.percentile(param_vals, 75)

    low  = param_vals < q25
    high = param_vals > q75

    if low.sum() < 2 or high.sum() < 2:
        return np.nan

    diff = residuals[high].mean(axis=0) - residuals[low].mean(axis=0)
    return np.sqrt(np.mean(diff ** 2))


# ==============================================================
# Bootstrap
# ==============================================================

def bootstrap_rms(
    residuals,
    param_vals,
    n_boot=2000,
    seed=42,
):
    """
    Non-parametric bootstrap of R(p).

    Resamples *simulations* (rows) with replacement,
    preserving the correlation structure across CDF bins.

    Returns
    -------
    rms_obs : float
    boot_samples : ndarray of shape (n_boot,)
    """
    rng = np.random.default_rng(seed)
    n   = len(residuals)

    rms_obs = _rms_response(residuals, param_vals)

    boot_samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_samples[i] = _rms_response(
            residuals[idx], param_vals[idx]
        )

    return rms_obs, boot_samples


# ==============================================================
# Null (permutation) test
# ==============================================================

def null_rms(
    residuals,
    param_vals,
    n_null=2000,
    seed=123,
):
    """
    Permutation null distribution for R(p).

    Breaks the association between parameter values and
    residual vectors by shuffling parameter labels.  Under
    the null, R(p) should be consistent with zero (up to
    finite-sample noise).

    Returns
    -------
    rms_obs : float
    null_samples : ndarray of shape (n_null,)
    p_value : float   (fraction of null >= observed)
    """
    rng = np.random.default_rng(seed)

    rms_obs = _rms_response(residuals, param_vals)

    null_samples = np.empty(n_null)
    for i in range(n_null):
        perm = rng.permutation(param_vals)
        null_samples[i] = _rms_response(residuals, perm)

    p_value = np.mean(null_samples >= rms_obs)

    return rms_obs, null_samples, p_value


# ==============================================================
# Combined table
# ==============================================================

def sensitivity_table(
    residuals,
    theta,
    params=None,
    n_boot=2000,
    n_null=2000,
    ci_level=0.95,
    boot_seed=42,
    null_seed=123,
):
    """
    Produce a summary DataFrame with columns:

        parameter | R_obs | ci_lo | ci_hi | null_floor | p_value | significant

    Parameters
    ----------
    residuals : ndarray (n_sims, n_bins)
    theta     : DataFrame indexed by sim_id, columns = parameter names
    params    : list of parameter names (default: all columns)
    ci_level  : confidence level for bootstrap CI (default 0.95)

    Returns
    -------
    df : pandas DataFrame
    """
    if params is None:
        params = list(theta.columns)

    alpha = 1 - ci_level
    rows  = []

    for p in params:
        pv = theta[p].values

        rms_obs, boot = bootstrap_rms(
            residuals, pv,
            n_boot=n_boot, seed=boot_seed,
        )

        ci_lo = np.nanpercentile(boot, 100 * alpha / 2)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2))

        _, null, p_val = null_rms(
            residuals, pv,
            n_null=n_null, seed=null_seed,
        )

        null_95 = np.nanpercentile(null, 95)

        rows.append({
            "parameter":  p,
            "R_obs":      rms_obs,
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
            "null_floor": null_95,
            "p_value":    p_val,
            "significant": rms_obs > null_95,
        })

    return pd.DataFrame(rows)


# ==============================================================
# Publication figure
# ==============================================================

def plot_sensitivity(
    df,
    ax=None,
    title=None,
    figsize=(7, 4),
    color_sig="#1a1a2e",
    color_insig="#b0b0b0",
    color_null="#e74c3c",
):
    """
    Bar chart of R(p) with bootstrap error bars and
    a null-floor line.

    Parameters
    ----------
    df : DataFrame from sensitivity_table()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df = df.sort_values("R_obs", ascending=True).reset_index(drop=True)

    n = len(df)
    y = np.arange(n)

    colors = [
        color_sig if s else color_insig
        for s in df["significant"]
    ]

    err_lo = np.maximum(df["R_obs"] - df["ci_lo"], 0)
    err_hi = np.maximum(df["ci_hi"] - df["R_obs"], 0)

    ax.barh(
        y,
        df["R_obs"],
        xerr=[err_lo, err_hi],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
        capsize=3,
        error_kw=dict(lw=1.2, color="#333333"),
        zorder=3,
    )

    null_floor = df["null_floor"].median()
    ax.axvline(
        null_floor,
        color=color_null,
        ls="--",
        lw=1.5,
        zorder=2,
        label=f"95% null floor ({null_floor:.4f})",
    )

    # p-value annotations
    for i, row in df.iterrows():
        if row["p_value"] < 0.001:
            label = "p < 0.001"
        elif row["p_value"] < 0.01:
            label = f"p = {row['p_value']:.3f}"
        elif row["p_value"] < 0.05:
            label = f"p = {row['p_value']:.2f}"
        else:
            label = f"p = {row['p_value']:.2f}"

        ax.text(
            max(row["ci_hi"], row["R_obs"]) + 0.0003,
            i,
            label,
            va="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel(r"RMS response $R_p$")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    if title:
        ax.set_title(title, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ==============================================================
# Multi-selection comparison figure
# ==============================================================

def plot_multi_selection(
    tables,
    labels,
    focus_params=None,
    figsize=(8, 5),
):
    """
    Grouped bar chart comparing R(p) across selections.

    Parameters
    ----------
    tables : list of DataFrames from sensitivity_table()
    labels : list of str (e.g. ["M1e6", "M1e7", "L2"])
    focus_params : list of parameter names to show
                   (default: all in first table)
    """
    if focus_params is None:
        focus_params = list(tables[0]["parameter"])

    n_params = len(focus_params)
    n_sel    = len(tables)

    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / n_sel
    cmap  = plt.cm.Dark2

    for j, (df, lab) in enumerate(zip(tables, labels)):
        df_idx = df.set_index("parameter")
        vals   = [df_idx.loc[p, "R_obs"] for p in focus_params]
        ci_lo  = [max(df_idx.loc[p, "R_obs"] - df_idx.loc[p, "ci_lo"], 0) for p in focus_params]
        ci_hi  = [max(df_idx.loc[p, "ci_hi"] - df_idx.loc[p, "R_obs"], 0) for p in focus_params]

        x = np.arange(n_params) + j * width

        ax.bar(
            x, vals,
            width=width,
            yerr=[ci_lo, ci_hi],
            label=lab,
            color=cmap(j / max(n_sel - 1, 1)),
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            error_kw=dict(lw=1),
            zorder=3,
        )

    # null floor from first table as reference
    null_floor = tables[0]["null_floor"].median()
    ax.axhline(
        null_floor, color="#e74c3c", ls="--", lw=1.2,
        label="95% null floor", zorder=2,
    )

    ax.set_xticks(np.arange(n_params) + width * (n_sel - 1) / 2)
    ax.set_xticklabels(focus_params, rotation=30, ha="right")
    ax.set_ylabel(r"RMS response $R_p$")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ==============================================================
# Quick demo with synthetic data
# ==============================================================

if __name__ == "__main__":

    np.random.seed(0)
    n_sims = 500
    n_bins = 150  # 3 k-values × 50 radial bins

    # Fake parameters (LH-like uniform draws)
    theta = pd.DataFrame({
        "Omega_m": np.random.uniform(0.1, 0.5, n_sims),
        "sigma_8": np.random.uniform(0.6, 1.0, n_sims),
        "A_SN1":   np.random.uniform(0.25, 4.0, n_sims),
        "A_AGN1":  np.random.uniform(0.25, 4.0, n_sims),
        "A_SN2":   np.random.uniform(0.5, 2.0, n_sims),
        "A_AGN2":  np.random.uniform(0.5, 2.0, n_sims),
    })

    # Fake residuals: inject signal into Omega_m and A_SN1
    noise = np.random.normal(0, 0.01, (n_sims, n_bins))

    signal_Om   = 0.05 * np.outer(
        theta["Omega_m"] - theta["Omega_m"].mean(),
        np.random.randn(n_bins),
    )
    signal_ASN1 = 0.04 * np.outer(
        theta["A_SN1"] - theta["A_SN1"].mean(),
        np.random.randn(n_bins),
    )

    residuals = noise + signal_Om + signal_ASN1

    print("Running bootstrap + null (synthetic demo)...")
    print("=" * 55)

    params = [
        "Omega_m", "sigma_8",
        "A_SN1", "A_AGN1",
        "A_SN2", "A_AGN2",
    ]

    df = sensitivity_table(
        residuals, theta,
        params=params,
        n_boot=2000,
        n_null=2000,
    )

    print(df.to_string(index=False, float_format="%.5f"))
    print()

    fig, ax = plot_sensitivity(
        df,
        title="Synthetic demo — bootstrap + null test",
    )

    fig.savefig("demo_sensitivity.png", dpi=150)
    print("Saved: demo_sensitivity.png")