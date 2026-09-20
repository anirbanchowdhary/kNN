"""
Six presentation figures for the AGN kNN-CDF project, built from real
pipeline output.

Assumes you have already run the notebooks in the order the README
prescribes (01 -> 02 -> 05 -> 06 -> 07), so `config.OUTPUT_DIR` already
holds the cached .npz files this script globs for:

    agn_knn_snap<snap>_M<mass_cut>_top10.npz        (notebook 01)
    agn_knn_snap<snap>_M<mass_cut>_n<N>.npz          (notebook 02)
    cv_knn_snap<snap>_M<mass_cut>_n<N>.npz            (notebook 05)
    galaxy_knn_snap<snap>_M<mass_cut>_n<N>.npz         (notebook 06)
    galaxy_cv_knn_snap<snap>_M<mass_cut>_n<N>.npz       (notebook 07)

and one real snapshot file under `config.SIM_PATH` (for Figure 1's toy
picture of a handful of real AGN positions -- everything else here works
off the cached summaries alone). If a required file is missing, this
prints which notebook to run first and stops -- it does not fall back to
invented numbers.

Run from anywhere: `python presentation/make_presentation_plots.py`
Writes PNGs to presentation/figs/.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.data_io import find_snapshots, read_bh_catalog
from src.selection import select_brightest_n
from src.params import load_params, align_to_params
from src.abundance import remove_abundance
from src.sensitivity import sensitivity_table, summary_dataframe, infer_layout
from src.cv import per_bin_std
from src.fisher import (
    linear_response, drop_uninformative_bins, fisher_matrix, combine_fisher,
    marginalized_covariance, sub_covariance,
)
from src.plotting import plot_fisher_ellipse

from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

OUTPUT_DIR = config.OUTPUT_DIR

# Bootstrap/null-permutation reps for sensitivity_table -- matches the
# notebooks' own default. Drop to a few hundred for a fast dry run.
N_BOOT = 2000
N_NULL = 2000

# Okabe-Ito colorblind-safe categorical palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#999999"
INK = "#333333"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 12,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def savefig(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def require(pattern, notebook):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern!r} in {OUTPUT_DIR} -- run {notebook} first."
        )
    return matches[-1]


# =========================================================================
# Load everything up front
# =========================================================================
print("Loading parameter table and cached kNN-CDF runs...")

theta_all = load_params(config.PARAMS_FILE)

top_pct = int(round(config.TOP_FRACTION * 100))
top_path = require(
    f"{OUTPUT_DIR}/agn_knn_snap{config.SNAP}_M{config.MASS_CUT:.0e}_top{top_pct}.npz",
    "notebook 01 (01_agn_luminosity_knn_sensitivity.ipynb)",
)
top_data = np.load(top_path, allow_pickle=True)

agn_path = require(
    f"{OUTPUT_DIR}/agn_knn_snap{config.SNAP}_M{config.MASS_CUT:.0e}_n*.npz",
    "notebook 02 (02_fixed_density_agn_knn.ipynb)",
)
agn_data = np.load(agn_path, allow_pickle=True)
N_TARGET = int(agn_data["nbh"][0])
assert np.all(agn_data["nbh"] == N_TARGET), "AGN run is not fixed-N -- wrong file?"

agn_cv_path = require(
    f"{OUTPUT_DIR}/cv_knn_snap{config.SNAP}_M{config.MASS_CUT:.0e}_n*.npz",
    "notebook 05 (05_cv_noise_floor.ipynb)",
)
agn_cv_data = np.load(agn_cv_path, allow_pickle=True)

gal_path = require(
    f"{OUTPUT_DIR}/galaxy_knn_snap{config.SNAP}_M{config.GALAXY_MASS_CUT:.0e}_n*.npz",
    "notebook 06 (06_galaxy_agn_complementarity.ipynb)",
)
gal_data = np.load(gal_path, allow_pickle=True)

gal_cv_path = require(
    f"{OUTPUT_DIR}/galaxy_cv_knn_snap{config.SNAP}_M{config.GALAXY_MASS_CUT:.0e}_n*.npz",
    "notebook 07 (07_fisher_forecast.ipynb)",
)
gal_cv_data = np.load(gal_cv_path, allow_pickle=True)

print(f"AGN fixed-N run:  N_TARGET={N_TARGET}, {len(agn_data['sim_ids'])} sims")
print(f"Galaxy fixed-N run: N_TARGET={int(gal_data['nbh'][0])}, {len(gal_data['sim_ids'])} sims")

# Fixed-N residuals: nbh is constant by construction, so abundance
# regression is a no-op -- plain mean-centering, exactly as notebooks
# 05/06/07 do it.
agn_n_k, agn_n_r = infer_layout(agn_data["kvals"], agn_data["rgrid"])
agn_theta = align_to_params(agn_data["sim_ids"], theta_all)
agn_residuals = agn_data["summaries"] - agn_data["summaries"].mean(axis=0)

gal_n_k, gal_n_r = infer_layout(gal_data["kvals"], gal_data["rgrid"])
gal_theta = align_to_params(gal_data["sim_ids"], theta_all)
gal_residuals = gal_data["summaries"] - gal_data["summaries"].mean(axis=0)

# Top-fraction run: nbh varies per sim, so the abundance trend is real
# and must be regressed out first.
top_theta = align_to_params(top_data["sim_ids"], theta_all)
top_residuals = remove_abundance(top_data["summaries"], top_data["nbh"])
top_n_k, top_n_r = infer_layout(top_data["kvals"], top_data["rgrid"])

print("Running sensitivity_table for the AGN fixed-N run (all 6 parameters)...")
table_agn = sensitivity_table(
    agn_residuals, agn_theta, n_k=agn_n_k, rgrid=agn_data["rgrid"], kvals=agn_data["kvals"],
    params=config.ALL_PARAMS, n_boot=N_BOOT, n_null=N_NULL,
)
summary_agn = summary_dataframe(table_agn)

print("Running sensitivity_table for the galaxy fixed-N run (all 6 parameters)...")
table_gal = sensitivity_table(
    gal_residuals, gal_theta, n_k=gal_n_k, rgrid=gal_data["rgrid"], kvals=gal_data["kvals"],
    params=config.ALL_PARAMS, n_boot=N_BOOT, n_null=N_NULL,
)
summary_gal = summary_dataframe(table_gal)

print("Running sensitivity_table for the top-fraction run (Omega_m only)...")
table_top = sensitivity_table(
    top_residuals, top_theta, n_k=top_n_k, rgrid=top_data["rgrid"], kvals=top_data["kvals"],
    params=["Omega_m"], n_boot=N_BOOT, n_null=N_NULL,
)


# =========================================================================
# 1. What is the kNN-CDF? -- real AGN positions from one real simulation,
#    real per-sim CDF from the cached fixed-N summary
# =========================================================================
def fig1_schematic():
    sim_id = int(agn_data["sim_ids"][0])
    snapshot_path = find_snapshots(config.SIM_PATH, config.SNAP, prefix="LH_")
    snapshot_path = next(p for p in snapshot_path if f"LH_{sim_id}/" in p)

    catalog = read_bh_catalog(snapshot_path)
    mask = select_brightest_n(catalog["bh_mass"], catalog["bh_mdot"], config.MASS_CUT, N_TARGET)
    pos = catalog["pos"][mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # left: a thin slab through the box, projected to 2D, with two real
    # query points and their real k=2 nearest-neighbor radius
    slab = pos[np.abs(pos[:, 2] - pos[:, 2].mean()) < config.BOXSIZE * 0.15]
    ax1.scatter(slab[:, 0], slab[:, 1], s=14, color=GRAY, zorder=2)

    tree = cKDTree(pos, boxsize=config.BOXSIZE)
    rng = np.random.default_rng(0)
    for color in (BLUE, ORANGE):
        q = rng.uniform(0, config.BOXSIZE, size=3)
        d, _ = tree.query(q, k=2)
        r_k = d[-1]  # distance to the 2nd-nearest AGN
        circle = plt.Circle((q[0], q[1]), r_k, fill=False, color=color, lw=2, zorder=3)
        ax1.add_patch(circle)
        ax1.scatter([q[0]], [q[1]], s=60, color=color, marker="*", zorder=4)

    ax1.set_xlim(0, config.BOXSIZE)
    ax1.set_ylim(0, config.BOXSIZE)
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"real AGN in LH_{sim_id}\n(thin slice, k=2 shown)")

    # right: this simulation's actual cached kNN-CDF, one line per k
    cdf = agn_data["summaries"][0].reshape(agn_n_k, agn_n_r)
    for ki, (k, color) in enumerate(zip(agn_data["kvals"], [BLUE, GREEN, ORANGE])):
        ax2.plot(agn_data["rgrid"], cdf[ki], color=color, lw=2.2, label=f"k={k}")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax2.set_ylabel("kNN-CDF")
    ax2.set_title(f"LH_{sim_id}'s actual kNN-CDF")
    ax2.legend()

    fig.suptitle("The kNN-CDF, from one real simulation", y=1.04)
    fig.tight_layout()
    savefig(fig, "1_schematic.png")


# =========================================================================
# 2. Headline result: which of the 6 parameters actually matter
# =========================================================================
def fig2_sensitivity_bar():
    df = summary_agn.set_index("parameter").loc[config.ALL_PARAMS].reset_index()
    order = np.argsort(df["R_obs"].values)
    df = df.iloc[order].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(df))
    colors = [BLUE if p in config.COSMO_PARAMS else ORANGE for p in df["parameter"]]
    alphas = [1.0 if s else 0.35 for s in df["significant"]]

    bars = ax.barh(y, df["R_obs"], color=colors, height=0.6, edgecolor="white", linewidth=0.5, zorder=3)
    for b, a in zip(bars, alphas):
        b.set_alpha(a)

    null_floor = df["null_floor"].median()
    ax.axvline(null_floor, color=INK, ls="--", lw=1.4, zorder=2)
    ax.text(null_floor, 0.5, "  chance level", fontsize=10, color=INK, va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel(r"parameter response, $R(p)$")
    ax.set_xlim(0, df["R_obs"].max() * 1.3)
    top_param = df.iloc[-1]["parameter"]
    ax.set_title(f"{top_param} is the standout parameter")

    handles = [Patch(color=BLUE, label="cosmological"), Patch(color=ORANGE, label="feedback")]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    savefig(fig, "2_sensitivity_bar.png")


# =========================================================================
# 3. Where the Omega_m signal lives, scale-by-scale
# =========================================================================
def fig3_scale_response():
    e = table_agn["Omega_m"]
    rgrid = agn_data["rgrid"]
    kvals = agn_data["kvals"]

    fig, ax = plt.subplots(figsize=(7, 4))
    null_band = np.nanmedian(e["null_floor"], axis=0)
    ax.fill_between(rgrid, -null_band, null_band, color=GRAY, alpha=0.3, lw=0, label="null band", zorder=1)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)

    for ki, (k, color) in enumerate(zip(kvals, [BLUE, GREEN, ORANGE])):
        ax.plot(rgrid, e["obs"][ki], color=color, lw=2.2, label=f"k={k}", zorder=3)
        ax.fill_between(rgrid, e["ci_lo"][ki], e["ci_hi"][ki], color=color, alpha=0.15, lw=0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax.set_ylabel("Omega_m response")
    ax.set_title(f"Omega_m's imprint vs. scale  (R={e['R_scalar']:.4f})")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "3_scale_response_omega_m.png")


# =========================================================================
# 4. Why the selection method mattered
# =========================================================================
def fig4_selection_comparison():
    r_top = table_top["Omega_m"]["R_scalar"]
    r_fixed = table_agn["Omega_m"]["R_scalar"]

    labels = [f"top {top_pct}% by\nluminosity", "fixed number\ndensity"]
    R = [r_top, r_fixed]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    bars = ax.bar(x, R, color=[GRAY, BLUE], width=0.55, edgecolor="white", zorder=3)

    for xi, r in zip(x, R):
        ax.text(xi, r + max(R) * 0.02, f"{r:.4f}", ha="center", fontsize=11, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Omega_m response, $R$")
    ax.set_ylim(0, max(R) * 1.35)
    ax.set_title("Fixing tracer density changed the signal")
    fig.tight_layout()
    savefig(fig, "4_selection_comparison.png")


# =========================================================================
# 5. Two tracers, two constraints
# =========================================================================
def fig5_tracer_comparison():
    params = ["Omega_m", "sigma_8"]
    sa = summary_agn.set_index("parameter").loc[params]
    sg = summary_gal.set_index("parameter").loc[params]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(params))
    w = 0.32

    b1 = ax.bar(x - w / 2, sa["R_obs"], width=w, color=BLUE, label="AGN", zorder=3)
    b2 = ax.bar(x + w / 2, sg["R_obs"], width=w, color=ORANGE, label="galaxies", zorder=3)
    for bars, sig in [(b1, sa["significant"]), (b2, sg["significant"])]:
        for b, s in zip(bars, sig):
            b.set_alpha(1.0 if s else 0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel(r"response, $R$")
    ax.set_title("Do the two tracers agree?\n(faded = not significant)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "5_tracer_comparison.png")


# =========================================================================
# 6. Putting it together: Fisher forecast ellipses
# =========================================================================
def fig6_fisher_ellipses():
    agn_cv_n_k, agn_cv_n_r = infer_layout(agn_cv_data["kvals"], agn_cv_data["rgrid"])
    gal_cv_n_k, gal_cv_n_r = infer_layout(gal_cv_data["kvals"], gal_cv_data["rgrid"])

    agn_cv_var = per_bin_std(agn_cv_data["summaries"], agn_cv_n_k, agn_cv_n_r).flatten() ** 2
    gal_cv_var = per_bin_std(gal_cv_data["summaries"], gal_cv_n_k, gal_cv_n_r).flatten() ** 2

    D_agn = linear_response(agn_residuals, agn_theta, params=config.ALL_PARAMS)
    D_gal = linear_response(gal_residuals, gal_theta, params=config.ALL_PARAMS)

    D_agn_kept, agn_cv_var_kept, _ = drop_uninformative_bins(D_agn, agn_cv_var)
    D_gal_kept, gal_cv_var_kept, _ = drop_uninformative_bins(D_gal, gal_cv_var)

    F_agn = fisher_matrix(D_agn_kept, agn_cv_var_kept)
    F_gal = fisher_matrix(D_gal_kept, gal_cv_var_kept)
    F_combined = combine_fisher(F_agn, F_gal)

    cov_agn = marginalized_covariance(F_agn)
    cov_gal = marginalized_covariance(F_gal)
    cov_combined = marginalized_covariance(F_combined)

    sub_agn = sub_covariance(cov_agn, config.ALL_PARAMS, ["Omega_m", "sigma_8"])
    sub_gal = sub_covariance(cov_gal, config.ALL_PARAMS, ["Omega_m", "sigma_8"])
    sub_combined = sub_covariance(cov_combined, config.ALL_PARAMS, ["Omega_m", "sigma_8"])

    for name, sub in [("AGN", sub_agn), ("galaxy", sub_gal), ("combined (naive)", sub_combined)]:
        print(f"{name:>16}: sigma(Omega_m)={np.sqrt(sub[0, 0]):.4f}  sigma(sigma_8)={np.sqrt(sub[1, 1]):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    center = (theta_all["Omega_m"].mean(), theta_all["sigma_8"].mean())

    plot_fisher_ellipse(ax, center, sub_agn, n_sigma=(1,), color=BLUE, label="AGN")
    plot_fisher_ellipse(ax, center, sub_gal, n_sigma=(1,), color=ORANGE, label="galaxy")
    plot_fisher_ellipse(ax, center, sub_combined, n_sigma=(1,), color=GREEN, label="combined (naive)")

    ax.plot(*center, marker="+", color=INK, ms=10, mew=1.5)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\sigma_8$")
    ax.set_title("Combining tracers tightens the forecast\n(1-sigma; no cross-covariance measured yet)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "6_fisher_ellipses.png")


if __name__ == "__main__":
    fig1_schematic()
    fig2_sensitivity_bar()
    fig3_scale_response()
    fig4_selection_comparison()
    fig5_tracer_comparison()
    fig6_fisher_ellipses()
