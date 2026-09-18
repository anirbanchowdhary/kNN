"""
generate_2pcf.py
================

Two-point correlation function ξ(r) for SMBH populations
in the CAMELS-IllustrisTNG LH suite.

Mirrors the kNN pipeline (generate_knn.py / generate_knn_active.py)
so that the same downstream analysis (abundance removal, bootstrap
sensitivity) can be applied to both summary statistics.

Usage
-----
    from src.generate_2pcf import run_suite

    # Mass-selected (like generate_knn.py)
    run_suite(snap=50, selection="mass", mass_cut=1e6)

    # Luminosity top-2% (like generate_knn_active.py)
    run_suite(snap=50, selection="luminosity",
              mass_cut=1e6, top_fraction=0.02)
"""

import numpy as np
import h5py
import hdf5plugin

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial import cKDTree

import os


# ==============================================================
# Constants (same as generate_knn_active.py)
# ==============================================================

MSUN    = 1.98847e33
YEAR    = 3.15576e7
C       = 2.99792458e10
EPS_RAD = 0.1


# ==============================================================
# 2PCF estimator for periodic box
# ==============================================================

def tpcf_periodic(pos, boxsize, r_edges):
    """
    Natural estimator ξ(r) = DD/RR - 1 for a periodic box.

    Uses cKDTree.count_neighbors for exact pair counts
    with periodic boundary conditions.  RR is analytic.

    Parameters
    ----------
    pos     : ndarray (N, 3)
    boxsize : float
    r_edges : ndarray (n_bins+1,)   bin edges in Mpc/h

    Returns
    -------
    xi : ndarray (n_bins,)
    """
    nbh = len(pos)
    if nbh < 2:
        return None

    tree = cKDTree(pos, boxsize=boxsize)

    # count_neighbors gives cumulative pair counts (including
    # self-pairs and double-counting).  diff gives shell counts.
    cumul = tree.count_neighbors(tree, r_edges)
    DD    = np.diff(cumul).astype(np.float64)

    # Analytic RR for a uniform Poisson process.
    # count_neighbors double-counts (i,j)+(j,i), so
    # the normalisation uses N*(N-1) not N*(N-1)/2.
    V_box   = boxsize ** 3
    V_shell = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    RR = nbh * (nbh - 1) * V_shell / V_box
    RR = nbh * (nbh - 1) * V_shell / V_box

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR > 0, DD / RR - 1.0, 0.0)

    return xi


# ==============================================================
# Worker
# ==============================================================
def find_snapshots(sim_path, snap):
    snap_name = f"snapshot_{snap:03d}.hdf5"
    with os.scandir(sim_path) as top:
        return sorted(
            os.path.join(e.path, snap_name)
            for e in top
            if e.is_dir(follow_symlinks=False) and e.name.startswith("LH_")
        )
        
def _process_snapshot(args):

    (
        snapshot,
        selection,
        mass_cut,
        top_fraction,
        min_bh,
        boxsize,
        r_edges,
    ) = args

    try:

        with h5py.File(snapshot, "r") as f:

            if "PartType5" not in f:
                return None

            h = f["Header"].attrs["HubbleParam"]

            pos = (
                f["PartType5"]["Coordinates"][:]
                .astype(np.float64)
                / 1e3                       # ckpc/h → cMpc/h
            )

            bh_mass = (
                f["PartType5"]["BH_Mass"][:]
                .astype(np.float64)
                * 1e10 / h                  # code → M_sun
            )

            bh_mdot = (
                f["PartType5"]["BH_Mdot"][:]
                .astype(np.float64)
            )

        # ======================================================
        # Derived quantities
        # ======================================================

        mdot_cgs = bh_mdot * 10.22 * MSUN / YEAR
        Lbol     = EPS_RAD * mdot_cgs * C ** 2
        Ledd     = 1.26e38 * bh_mass
        fedd     = Lbol / Ledd

        # ======================================================
        # Selection
        # ======================================================

        base = bh_mass > mass_cut

        if np.sum(base) < 20:
            return None

        if selection == "mass":
            mask = base

        elif selection == "luminosity":
            good = base & np.isfinite(Lbol) & (Lbol > 0)
            if np.sum(good) < 20:
                return None
            Lcut = np.percentile(Lbol[good], 100 * (1 - top_fraction))
            mask = good & (Lbol >= Lcut)

        elif selection == "fedd":
            good = base & np.isfinite(fedd) & (fedd > 0)
            if np.sum(good) < 20:
                return None
            fcut = np.percentile(fedd[good], 100 * (1 - top_fraction))
            mask = good & (fedd >= fcut)

        else:
            raise ValueError(f"Unknown selection={selection}")

        pos = pos[mask]
        nbh = len(pos)

        if nbh < min_bh:
            return None

        # ======================================================
        # 2PCF
        # ======================================================

        xi = tpcf_periodic(pos, boxsize, r_edges)

        if xi is None:
            return None

        sim_id = int(
            snapshot.split("LH_")[1].split("/")[0]
        )

        return (sim_id, xi, nbh)

    except Exception as e:
        print(f"\nFAILED: {snapshot}\n{e}")
        return None


# ==============================================================
# Main
# ==============================================================

def run_suite(
    snap=50,
    selection="mass",
    mass_cut=1e6,
    top_fraction=0.10,
    min_bh=5,
    boxsize=25.0,
    n_bins=50,
    r_edges=None,
    nproc=None,
    sim_path="../../Data/Sims/IllustrisTNG/LH",
    output_dir="../outputs",
):
    """
    Compute ξ(r) for every LH simulation and save to .npz.

    The output format matches generate_knn.py:
        sim_ids, summaries, nbh, rgrid (= bin centres)
    so the same downstream pipeline works unchanged.
    """

    if r_edges is None:
        r_edges = np.logspace(-1.5, 1.2, n_bins + 1)

    if nproc is None:
        nproc = min(16, os.cpu_count())

    # Bin centres (geometric mean) for downstream plotting
    r_centres = np.sqrt(r_edges[:-1] * r_edges[1:])

    files = find_snapshots(sim_path, snap)

    print(f"Found {len(files)} files")

    worker_args = [
        (
            fname,
            selection,
            mass_cut,
            top_fraction,
            min_bh,
            boxsize,
            r_edges,
        )
        for fname in files
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
    nbh       = np.array([r[2] for r in results])

    # ----------------------------------------------------------
    # Output filename mirrors the kNN convention but with
    # a "2pcf_" prefix so both coexist in the same directory.
    # ----------------------------------------------------------

    sel_tag = "" if selection == "mass" else f"{selection}_"

    outfile = (
        f"{output_dir}/"
        f"2pcf_{sel_tag}"
        f"snap{snap}_"
        f"M{mass_cut:.0e}.npz"
    )

    np.savez(
        outfile,
        sim_ids=sim_ids,
        summaries=summaries,
        nbh=nbh,
        rgrid=r_centres,
        r_edges=r_edges,
        snap=snap,
        selection=selection,
        mass_cut=mass_cut,
        top_fraction=top_fraction,
    )

    print()
    print(f"Saved: {outfile}")
    print(f"Successful runs = {len(results)}")
    print(f"Summary shape   = {summaries.shape}")
    print(f"Mean N_BH       = {np.mean(nbh):.1f}")

    return {
        "sim_ids":   sim_ids,
        "summaries": summaries,
        "nbh":       nbh,
        "rgrid":     r_centres,
    }