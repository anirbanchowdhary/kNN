"""
generate_knn_galaxies.py
========================

kNN-CDF summaries for CAMELS galaxies (subhalos) from the
SUBFIND group catalogues.

Selections
----------
- "mass":  stellar-mass cuts at 10^9, 10^10, 10^11 Msun
- "ssfr":  top fraction in sSFR = SFR / M_star

Mirrors the SMBH pipeline exactly so the downstream modules
(remove_abundance, sensitivity_bootstrap, selection_bias) work
unchanged on the outputs.

Output filename convention (parallel to SMBH outputs):
  knn_gal_snap{snap}_Mstar{cut:.0e}.npz                (mass)
  knn_gal_ssfr_snap{snap}_Mstar{cut:.0e}.npz           (sSFR top-X%)
"""

import os
import numpy as np
import h5py
import hdf5plugin

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


# ==============================================================
# Constants
# ==============================================================

MIN_STAR_PARTICLES = 20   # resolution cut: subhalos with fewer
                          # than this many star particles are
                          # discarded as numerical artefacts


# ==============================================================
# Helpers
# ==============================================================

def cdf_on_grid(distances, rgrid):
    rs = np.sort(distances)
    Fs = np.arange(1, len(rs) + 1) / len(rs)
    return interp1d(
        rs, Fs,
        bounds_error=False,
        fill_value=(0, 1),
    )(rgrid)


def _find_subfind_file(sim_dir, snap):
    """
    Locate the SUBFIND group catalogue for a snapshot.

    Tries the post-2024 CAMELS single-file convention first,
    then falls back to multi-file and legacy naming.
    """
    candidates = [
        f"{sim_dir}/fof_subhalo_tab_{snap:03d}.hdf5",
        f"{sim_dir}/groups_{snap:03d}.hdf5",
        f"{sim_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.hdf5",
        f"{sim_dir}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.0.hdf5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _read_subhalos(path):
    """
    Read positions, stellar masses, SFRs, and star particle
    counts from a SUBFIND catalogue.

    Handles single-file catalogues only; for multi-file
    catalogues the caller must concatenate.
    """
    with h5py.File(path, "r") as f:

        if "Subhalo" not in f:
            return None

        h = f["Header"].attrs["HubbleParam"]

        sub_pos       = f["Subhalo"]["SubhaloPos"][:].astype(np.float64)
        sub_mass_type = f["Subhalo"]["SubhaloMassType"][:].astype(np.float64)
        sub_sfr       = f["Subhalo"]["SubhaloSFR"][:].astype(np.float64)

        # Star particle count for resolution cut (if present)
        if "SubhaloLenType" in f["Subhalo"]:
            sub_nstar = f["Subhalo"]["SubhaloLenType"][:, 4].astype(np.int64)
        else:
            sub_nstar = None

    pos     = sub_pos / 1e3                            # ckpc/h -> cMpc/h
    Mstar   = sub_mass_type[:, 4] * 1e10 / h           # -> Msun
    SFR     = sub_sfr                                  # already Msun/yr
    return pos, Mstar, SFR, sub_nstar, h


# ==============================================================
# Worker
# ==============================================================

def _process_snapshot(args):

    (
        sim_dir,
        snap,
        selection,
        mass_cut,
        top_fraction,
        min_galaxies,
        boxsize,
        kvals,
        rgrid,
        random_pos,
    ) = args

    try:
        path = _find_subfind_file(sim_dir, snap)
        if path is None:
            return None

        out = _read_subhalos(path)
        if out is None:
            return None
        pos, Mstar, SFR, nstar, h = out

        # ----------------------------------------------------------
        # Resolution + sanity cut
        # ----------------------------------------------------------
        finite = (
            np.isfinite(Mstar)
            & (Mstar > 0)
            & np.isfinite(SFR)
            & np.isfinite(pos).all(axis=1)
        )
        if nstar is not None:
            finite &= (nstar >= MIN_STAR_PARTICLES)

        if finite.sum() < 20:
            return None

        pos   = pos[finite]
        Mstar = Mstar[finite]
        SFR   = SFR[finite]

        # ----------------------------------------------------------
        # Base stellar-mass cut
        # ----------------------------------------------------------
        base = Mstar > mass_cut

        if base.sum() < 20:
            return None

        # ----------------------------------------------------------
        # Selection
        # ----------------------------------------------------------
        if selection == "mass":
            mask = base

        elif selection == "ssfr":
            with np.errstate(divide="ignore", invalid="ignore"):
                sSFR = np.where(Mstar > 0, SFR / Mstar, 0.0)

            good = base & np.isfinite(sSFR) & (sSFR > 0)
            if good.sum() < 20:
                return None

            cut = np.percentile(
                sSFR[good], 100 * (1 - top_fraction)
            )
            mask = good & (sSFR >= cut)

        else:
            raise ValueError(f"Unknown selection={selection}")

        pos = pos[mask]
        ngal = len(pos)
        if ngal < min_galaxies:
            return None

        # ----------------------------------------------------------
        # kNN
        # ----------------------------------------------------------
        # Wrap positions into [0, boxsize) for cKDTree
        pos = pos % boxsize

        tree = cKDTree(pos, boxsize=boxsize)
        dist, _ = tree.query(random_pos, k=kvals)

        outputs = []
        for i in range(len(kvals)):
            outputs.append(cdf_on_grid(dist[:, i], rgrid))

        summary = np.concatenate(outputs)

        sim_id = int(
            sim_dir.split("LH_")[1].split("/")[0]
        )
        return (sim_id, summary, ngal)

    except Exception as e:
        print(f"\nFAILED: {sim_dir}\n  {type(e).__name__}: {e}")
        return None


# ==============================================================
# Main
# ==============================================================

def run_suite(
    snap=50,
    selection="mass",
    mass_cut=1e9,
    top_fraction=0.10,
    min_galaxies=5,
    nrand=100000,
    kvals=(1, 2, 4),
    boxsize=25.0,
    rgrid=None,
    nproc=None,
    sim_path="../../Data/Sims/IllustrisTNG/LH",
    output_dir="../outputs",
    random_seed=42,
):
    """
    Compute kNN-CDFs for galaxies in every LH simulation.

    Parameters
    ----------
    selection : "mass" or "ssfr"
    mass_cut  : stellar-mass threshold in Msun
                (defines the base sample for both selections)
    top_fraction : for "ssfr", the top fraction by sSFR to keep
    """
    if rgrid is None:
        rgrid = np.logspace(-1.5, 1.2, 50)
    if nproc is None:
        nproc = min(16, os.cpu_count())

    np.random.seed(random_seed)
    random_pos = np.random.uniform(0, boxsize, size=(nrand, 3))

    sim_dirs = sorted(glob(f"{sim_path}/LH_*"))
    print(f"Found {len(sim_dirs)} simulation directories")

    worker_args = [
        (
            d, snap, selection, mass_cut, top_fraction,
            min_galaxies, boxsize, kvals, rgrid, random_pos,
        )
        for d in sim_dirs
    ]

    with Pool(nproc) as pool:
        results = list(
            tqdm(
                pool.imap(_process_snapshot, worker_args),
                total=len(worker_args),
            )
        )

    results = [r for r in results if r is not None]

    sim_ids   = np.array([r[0] for r in results])
    summaries = np.array([r[1] for r in results])
    ngal      = np.array([r[2] for r in results])

    # ----------------------------------------------------------
    # Output filename
    # ----------------------------------------------------------
    if selection == "mass":
        sel_tag = ""
    else:
        pct = int(round(top_fraction * 100))
        sel_tag = f"{selection}{pct}_"
    outfile = (
        f"{output_dir}/"
        f"knn_gal_{sel_tag}"
        f"snap{snap}_"
        f"Mstar{mass_cut:.0e}.npz"
    )

    # Save with key 'nbh' for compatibility with downstream
    # remove_abundance() that expects that field name; we
    # also save it as 'ngal' for clarity.
    np.savez(
        outfile,
        sim_ids=sim_ids,
        summaries=summaries,
        nbh=ngal,          # for compatibility w/ existing pipeline
        ngal=ngal,
        rgrid=rgrid,
        snap=snap,
        selection=selection,
        mass_cut=mass_cut,
        top_fraction=top_fraction,
        kvals=np.array(kvals),
    )

    print()
    print(f"Saved: {outfile}")
    print(f"Successful runs = {len(results)}")
    print(f"Summary shape   = {summaries.shape}")
    print(f"Mean N_gal      = {np.mean(ngal):.1f}")

    return {
        "sim_ids":   sim_ids,
        "summaries": summaries,
        "nbh":       ngal,
        "rgrid":     rgrid,
    }