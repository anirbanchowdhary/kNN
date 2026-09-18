# src/generate_knn_active.py

import numpy as np
import h5py
import hdf5plugin

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

import os


# ==========================================================
# Constants
# ==========================================================

MSUN = 1.98847e33
YEAR = 3.15576e7
C = 2.99792458e10

EPS_RAD = 0.1


# ==========================================================
# Helper
# ==========================================================

def cdf_on_grid(distances, rgrid):

    rs = np.sort(distances)

    Fs = np.arange(
        1,
        len(rs)+1
    )/len(rs)

    return interp1d(
        rs,
        Fs,
        bounds_error=False,
        fill_value=(0,1)
    )(rgrid)


# ==========================================================
# Worker
# ==========================================================

def _process_snapshot(args):

    (
        snapshot,
        selection,
        mass_cut,
        top_fraction,
        boxsize,
        kvals,
        rgrid,
        random_pos
    ) = args

    try:

        with h5py.File(snapshot,'r') as f:

            if "PartType5" not in f:
                return None

            h = f["Header"].attrs["HubbleParam"]

            pos = (
                f["PartType5"]["Coordinates"][:]
                .astype(np.float64)
                / 1e3
            )

            bh_mass = (
                f["PartType5"]["BH_Mass"][:]
                .astype(np.float64)
                * 1e10
                / h
            )

            bh_mdot = (
                f["PartType5"]["BH_Mdot"][:]
                .astype(np.float64)
            )

        # ==================================================
        # Luminosity
        # ==================================================

        mdot_msunyr = bh_mdot * 10.22

        mdot_cgs = (
            mdot_msunyr
            * MSUN
            / YEAR
        )

        Lbol = (
            EPS_RAD
            * mdot_cgs
            * C**2
        )

        Ledd = (
            1.26e38
            * bh_mass
        )

        fedd = Lbol / Ledd

        # ==================================================
        # Base mass cut
        # ==================================================

        base = (
            bh_mass > mass_cut
        )

        if np.sum(base) < 20:
            return None

        # ==================================================
        # Selection
        # ==================================================

        if selection == "mass":

            mask = base

        elif selection == "luminosity":

            good = (
                base
                &
                np.isfinite(Lbol)
                &
                (Lbol > 0)
            )

            if np.sum(good) < 20:
                return None

            Lcut = np.percentile(
                Lbol[good],
                100*(1-top_fraction)
            )

            mask = (
                good
                &
                (Lbol >= Lcut)
            )

        elif selection == "fedd":

            good = (
                base
                &
                np.isfinite(fedd)
                &
                (fedd > 0)
            )

            if np.sum(good) < 20:
                return None

            fcut = np.percentile(
                fedd[good],
                100*(1-top_fraction)
            )

            mask = (
                good
                &
                (fedd >= fcut)
            )

        else:

            raise ValueError(
                f"Unknown selection={selection}"
            )

        pos = pos[mask]

        nbh = len(pos)

        if nbh < max(kvals)+5:
            return None

        # ==================================================
        # kNN
        # ==================================================

        tree = cKDTree(
            pos,
            boxsize=boxsize
        )

        dist,_ = tree.query(
            random_pos,
            k=kvals
        )

        outputs = []

        for i in range(len(kvals)):

            outputs.append(
                cdf_on_grid(
                    dist[:,i],
                    rgrid
                )
            )

        summary = np.concatenate(
            outputs
        )

        sim_id = int(
            snapshot.split("LH_")[1]
                    .split("/")[0]
        )

        return (
            sim_id,
            summary,
            nbh
        )

    except Exception:

        return None


# ==========================================================
# Main
# ==========================================================

def run_suite(
    snap=50,
    selection="mass",
    mass_cut=1e6,
    top_fraction=0.10,
    nrand=100000,
    kvals=(1,2,4),
    boxsize=25.0,
    rgrid=None,
    nproc=16,
    sim_path="../../Data/Sims/IllustrisTNG/LH",
    output_dir="../outputs",
    random_seed=42
):

    if rgrid is None:

        rgrid = np.logspace(
            -1.5,
            1.2,
            50
        )

    np.random.seed(
        random_seed
    )

    random_pos = np.random.uniform(
        0,
        boxsize,
        size=(nrand,3)
    )

    files = sorted(
        glob(
            f"{sim_path}/LH_*/snapshot_{snap:03d}.hdf5"
        )
    )

    print(
        f"Found {len(files)} files"
    )

    worker_args = [

        (
            fname,
            selection,
            mass_cut,
            top_fraction,
            boxsize,
            kvals,
            rgrid,
            random_pos
        )

        for fname in files

    ]

    with Pool(nproc) as pool:

        results = list(

            tqdm(

                pool.imap(
                    _process_snapshot,
                    worker_args
                ),

                total=len(worker_args)

            )

        )

    results = [
        r for r in results
        if r is not None
    ]

    sim_ids = np.array(
        [r[0] for r in results]
    )

    summaries = np.array(
        [r[1] for r in results]
    )

    nbh = np.array(
        [r[2] for r in results]
    )

    outfile = (
        f"{output_dir}/"
        f"knn_"
        f"{selection}_"
        f"snap{snap}_"
        f"M{mass_cut:.0e}.npz"
    )

    np.savez(

        outfile,

        sim_ids=sim_ids,
        summaries=summaries,
        nbh=nbh,

        rgrid=rgrid,

        snap=snap,

        selection=selection,

        mass_cut=mass_cut,

        top_fraction=top_fraction

    )

    print()
    print("Saved:")
    print(outfile)

    print()
    print(
        "Successful runs =",
        len(results)
    )

    print(
        "Summary shape =",
        summaries.shape
    )

    print(
        "Mean N_BH =",
        np.mean(nbh)
    )

    return {
        "sim_ids":sim_ids,
        "summaries":summaries,
        "nbh":nbh,
        "rgrid":rgrid
    }