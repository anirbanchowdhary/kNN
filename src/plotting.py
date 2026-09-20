"""
Figures for the sensitivity_table() output, and for the 1P parameter sweep.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_1p_sweep(rgrid, kvals, step_values, mean_cdfs, parameter_name,
                   figsize=None, logx=True):
    """
    One panel per k: the mean kNN-CDF at every 1P grid step for one
    parameter, colored by the parameter's value at that step. A different
    question from `plot_scale_response`'s quartile-contrast + null test --
    with marginalization noise gone (single parameter varied, others at
    fiducial), the first thing to check is simply whether the CDF moves
    smoothly/monotonically with the parameter at all.

    Parameters
    ----------
    rgrid          : (n_r,) radial bin centres
    kvals          : sequence of k values, length n_k
    step_values    : (n_steps,) the parameter's actual value at each step
                     (not just a step index -- read from the 1P params table)
    mean_cdfs      : (n_steps, n_k, n_r) mean kNN-CDF at each step
    parameter_name : for the title
    """
    n_k = len(kvals)
    if figsize is None:
        figsize = (4.2 * n_k, 3.4)

    fig, axes = plt.subplots(1, n_k, figsize=figsize, squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("viridis")
    vmin, vmax = np.min(step_values), np.max(step_values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for ki in range(n_k):
        ax = axes[ki]
        for i, v in enumerate(step_values):
            ax.plot(rgrid, mean_cdfs[i, ki], color=cmap(norm(v)), lw=1.4)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
        ax.set_title(f"k={kvals[ki]}")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("mean kNN-CDF")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=parameter_name, fraction=0.05, pad=0.02)

    fig.suptitle(f"1P sweep: {parameter_name}", y=1.05)
    return fig, axes


def plot_scale_response(table, parameter, ax=None, figsize=(7, 4), logx=True):
    """
    One parameter's scale-resolved response: a line per k, a shaded
    bootstrap CI band, and a symmetric null band. Where a colored line
    leaves the grey null band is where that parameter's imprint on the
    AGN kNN-CDF exceeds chance, at that scale and that k.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    e = table[parameter]
    rgrid = table["rgrid"]
    kvals = table["kvals"]
    n_k = table["n_k"]

    cmap = plt.get_cmap("viridis")
    for ki in range(n_k):
        colour = cmap(ki / max(n_k - 1, 1))
        ax.plot(rgrid, e["obs"][ki], color=colour, lw=1.6, label=f"k={kvals[ki]}", zorder=3)
        ax.fill_between(rgrid, e["ci_lo"][ki], e["ci_hi"][ki], color=colour, alpha=0.18, lw=0, zorder=2)

    null_band = np.nanmedian(e["null_floor"], axis=0)
    ax.fill_between(rgrid, -null_band, null_band, color="0.6", alpha=0.25, lw=0, zorder=1, label="95% null band")
    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel(r"$\langle x \rangle_{\rm high} - \langle x \rangle_{\rm low}$")
    ax.set_title(f"{parameter}  ($R={e['R_scalar']:.4f}$, p={e['R_p_value']:.3f})")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_scale_grid(table, params=None, ncols=3, figsize=None, logx=True, share_y=False):
    """Small-multiples grid of `plot_scale_response`, one panel per parameter."""
    if params is None:
        params = table["params"]

    n = len(params)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=share_y)
    axes_flat = axes.ravel()

    for j, p in enumerate(params):
        plot_scale_response(table, p, ax=axes_flat[j], logx=logx)
        if j != 0 and axes_flat[j].get_legend() is not None:
            axes_flat[j].get_legend().remove()

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_sensitivity_bar(summary_df, cosmo_params, astro_params, ax=None, figsize=(7.5, 4)):
    """
    Bar chart ranking all parameters by scalar R(p), colored by whether the
    parameter is cosmological or astrophysical (feedback).

    The dashed line is the 95% permutation-null floor -- the R a parameter
    reaches by chance alone. That line, not the error bars, is what decides
    significance: R is an RMS and so is positive-definite, meaning its
    bootstrap CI excludes zero even for a parameter that does nothing.
    Parameters that don't clear FDR-corrected significance are drawn faded,
    and each bar is annotated with its q-value.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df = summary_df.set_index("parameter").loc[cosmo_params + astro_params].reset_index()
    y = np.arange(len(df))

    colors = ["#1a1a2e" if p in cosmo_params else "#e07a5f" for p in df["parameter"]]
    alphas = [1.0 if s else 0.35 for s in df["significant"]]
    err_lo = np.maximum(df["R_obs"] - df["ci_lo"], 0)
    err_hi = np.maximum(df["ci_hi"] - df["R_obs"], 0)

    bars = ax.barh(y, df["R_obs"], xerr=[err_lo, err_hi], color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6, capsize=3,
                   error_kw=dict(lw=1.2, color="#333333"), zorder=3)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    null_floor = df["null_floor"].median()
    ax.axvline(null_floor, color="#c0392b", ls="--", lw=1.5, zorder=4,
               label=f"95% null floor ({null_floor:.4f})")

    for i, row in df.iterrows():
        q = row["q_value"]
        label = "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
        ax.text(max(row["ci_hi"], row["R_obs"]) * 1.03, i, label,
                va="center", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel(r"RMS response $R(p)$")
    ax.set_xlim(0, max(df["ci_hi"].max(), null_floor) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles = [
        Patch(color="#1a1a2e", label="cosmological"),
        Patch(color="#e07a5f", label="astrophysical (feedback)"),
        *ax.get_legend_handles_labels()[0],
    ]
    # upper right: the largest bar sits at the bottom, so the top rows are clear
    ax.legend(handles=handles, fontsize=8, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    return fig, ax


def plot_fisher_ellipse(ax, center, cov2x2, n_sigma=(1, 2), color="#1a1a2e", label=None, alpha=0.25):
    """
    Draw `n_sigma` Fisher confidence ellipses for a 2-parameter
    covariance, centered at `center` (e.g. the LH suite's mean/fiducial
    parameter values). Draws the largest sigma first so smaller ones
    layer on top and stay visible; only the innermost ellipse carries
    `label`, so a legend shows one entry per forecast, not one per sigma.
    """
    from matplotlib.patches import Ellipse

    from .fisher import confidence_ellipse_params

    sigmas = sorted(n_sigma, reverse=True)
    for i, ns in enumerate(sigmas):
        width, height, angle = confidence_ellipse_params(cov2x2, n_sigma=ns)
        ell = Ellipse(
            center, width, height, angle=angle,
            facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.4,
            label=label if ns == min(sigmas) else None,
            zorder=2 + i,
        )
        ax.add_patch(ell)
    return ax


def plot_bin_correlation(rgrid, kvals, corr, ax=None, figsize=(7, 4), logx=True):
    """
    `complementarity.bin_correlation`'s (n_k, n_r) output, one line per k.

    Near +-1: the two tracers' fluctuations move together at that scale
    (redundant). Near 0: independent (complementary). Unlike
    `plot_scale_response`, there's no null band here -- this is a
    descriptive correlation, not a significance test against a parameter.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_k = len(kvals)
    cmap = plt.get_cmap("viridis")
    for ki in range(n_k):
        colour = cmap(ki / max(n_k - 1, 1))
        ax.plot(rgrid, corr[ki], color=colour, lw=1.6, label=f"k={kvals[ki]}")

    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel("per-bin correlation (across sims)")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
