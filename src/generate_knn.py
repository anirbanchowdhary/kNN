# src/generate_knn.py

import numpy as np
import h5py
import hdf5plugin

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

import os


# ============================================================
# Helper
# ============================================================

def cdf_on_grid(distances, rgrid):

    rs = np.sort(distances)

    Fs = np.arange(
        1,
        len(rs)+1
    ) / len(rs)

    return interp1d(
        rs,
        Fs,
        bounds_error=False,
        fill_value=(0,1)
    )(rgrid)


# ============================================================
# Worker
# ============================================================

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
        mcut,
        boxsize,
        kvals,
        rgrid,
        random_pos
    ) = args

    try:

        with h5py.File(snapshot, "r") as f:

            if "PartType5" not in f:
                return None

            h = f["Header"].attrs["HubbleParam"]

            pos = (
                f["PartType5"]["Coordinates"][:]
                / 1e3
            )

            mass = (
                f["PartType5"]["BH_Mass"][:]
                * 1e10 / h
            )

        # ----------------------------
        # BH mass cut
        # ----------------------------

        mask = mass > mcut

        pos = pos[mask]

        nbh = len(pos)

        if nbh < max(kvals) + 5:
            return None

        # ----------------------------
        # KD tree
        # ----------------------------

        tree = cKDTree(
            pos,
            boxsize=boxsize
        )

        dist, _ = tree.query(
            random_pos,
            k=kvals
        )

        outputs = []

        for i in range(len(kvals)):

            outputs.append(
                cdf_on_grid(
                    dist[:, i],
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

    except Exception as e:

        print(
            f"\nFAILED: {snapshot}"
        )

        print(e)

        return None


# ============================================================
# Main function
# ============================================================

def run_suite(
    snap=50,
    mcut=1e6,
    nrand=100000,
    kvals=(1,2,4),
    boxsize=25.0,
    rgrid=None,
    nproc=None,
    sim_path="../../Data/Sims/IllustrisTNG/LH",
    output_dir="../outputs",
    random_seed=42,
):

    if rgrid is None:

        rgrid = np.logspace(
            -1.5,
            1.2,
            50
        )

    if nproc is None:

        nproc = min(
            16,
            os.cpu_count()
        )

    np.random.seed(
        random_seed
    )

    random_pos = np.random.uniform(
        0,
        boxsize,
        size=(nrand,3)
    )

    files = find_snapshots(sim_path, snap)


    print(
        f"Found {len(files)} files"
    )

    worker_args = [

        (
            fname,
            mcut,
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
        f"knn_snap{snap}_"
        f"M{mcut:.0e}.npz"
    )

    np.savez(

        outfile,

        sim_ids=sim_ids,
        summaries=summaries,
        nbh=nbh,

        rgrid=rgrid,

        snap=snap,
        mcut=mcut,

        kvals=np.array(kvals)

    )

    print()
    print("Saved:")
    print(outfile)

    print()
    print("Successful runs =", len(results))
    print("Summary shape =", summaries.shape)
    print("Mean N_BH =", np.mean(nbh))

    return {

        "sim_ids": sim_ids,
        "summaries": summaries,
        "nbh": nbh,
        "rgrid": rgrid

    }