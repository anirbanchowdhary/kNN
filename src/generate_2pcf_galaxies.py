"""
generate_2pcf_galaxies.py
=========================

Two-point correlation function ξ(r) for CAMELS galaxies.
Mirrors generate_2pcf.py but operates on SUBFIND catalogues.
Used for the kNN-vs-ξ(r) comparison applied to galaxies.
"""

import os
import numpy as np
import h5py
import hdf5plugin

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial import cKDTree

from src.generate_knn_galaxies import (
    MIN_STAR_PARTICLES,
    _find_subfind_file,
    _read_subhalos,
)


# ==============================================================
# 2PCF estimator (identical to SMBH version)
# ==============================================================

def tpcf_periodic(pos, boxsize, r_edges):
    nbh = len(pos)
    if nbh < 2:
        return None

    pos = pos % boxsize
    tree = cKDTree(pos, boxsize=boxsize)

    cumul = tree.count_neighbors(tree, r_edges)
    DD = np.diff(cumul).astype(np.float64)

    V_box   = boxsize ** 3
    V_shell = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    RR = nbh * (nbh - 1) * V_shell / V_box

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR > 0, DD / RR - 1.0, 0.0)
    return xi


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
        r_edges,
    ) = args

    try:
        path = _find_subfind_file(sim_dir, snap)
        if path is None:
            return None

        out = _read_subhalos(path)
        if out is None:
            return None
        pos, Mstar, SFR, nstar, h = out

        finite = (
            np.isfinite(Mstar) & (Mstar > 0)
            & np.isfinite(SFR)
            & np.isfinite(pos).all(axis=1)
        )
        if nstar is not None:
            finite &= (nstar >= MIN_STAR_PARTICLES)
        if finite.sum() < 20:
            return None

        pos = pos[finite]; Mstar = Mstar[finite]; SFR = SFR[finite]

        base = Mstar > mass_cut
        if base.sum() < 20:
            return None

        if selection == "mass":
            mask = base
        elif selection == "ssfr":
            with np.errstate(divide="ignore", invalid="ignore"):
                sSFR = np.where(Mstar > 0, SFR / Mstar, 0.0)
            good = base & np.isfinite(sSFR) & (sSFR > 0)
            if good.sum() < 20:
                return None
            cut = np.percentile(sSFR[good], 100 * (1 - top_fraction))
            mask = good & (sSFR >= cut)
        else:
            raise ValueError(f"Unknown selection={selection}")

        pos = pos[mask]
        ngal = len(pos)
        if ngal < min_galaxies:
            return None

        xi = tpcf_periodic(pos, boxsize, r_edges)
        if xi is None:
            return None

        sim_id = int(sim_dir.split("LH_")[1].split("/")[0])
        return (sim_id, xi, ngal)

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
    boxsize=25.0,
    n_bins=50,
    r_edges=None,
    nproc=None,
    sim_path="../../Data/Sims/IllustrisTNG/LH",
    output_dir="../outputs",
):
    if r_edges is None:
        r_edges = np.logspace(-1.5, 1.2, n_bins + 1)
    if nproc is None:
        nproc = min(16, os.cpu_count())

    r_centres = np.sqrt(r_edges[:-1] * r_edges[1:])

    sim_dirs = sorted(glob(f"{sim_path}/LH_*"))
    print(f"Found {len(sim_dirs)} simulation directories")

    worker_args = [
        (d, snap, selection, mass_cut, top_fraction,
         min_galaxies, boxsize, r_edges)
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

    if selection == "mass":
        sel_tag = ""
    else:
        pct = int(round(top_fraction * 100))
        sel_tag = f"{selection}{pct}_"
    outfile = (
        f"{output_dir}/"
        f"2pcf_gal_{sel_tag}"
        f"snap{snap}_"
        f"Mstar{mass_cut:.0e}.npz"
    )

    np.savez(
        outfile,
        sim_ids=sim_ids,
        summaries=summaries,
        nbh=ngal,
        ngal=ngal,
        rgrid=r_centres,
        r_edges=r_edges,
        snap=snap,
        selection=selection,
        mass_cut=mass_cut,
        top_fraction=top_fraction,
    )

    print(f"\nSaved: {outfile}")
    print(f"Successful runs = {len(results)}")
    print(f"Summary shape   = {summaries.shape}")
    print(f"Mean N_gal      = {np.mean(ngal):.1f}")

    return {
        "sim_ids":   sim_ids,
        "summaries": summaries,
        "nbh":       ngal,
        "rgrid":     r_centres,
    }