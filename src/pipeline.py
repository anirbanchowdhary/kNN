"""
End-to-end run: every LH simulation's snapshot -> one AGN kNN-CDF summary.

Parallelized across simulations with multiprocessing, since each
snapshot's kNN-CDF is independent of every other's.
"""

import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .config import (
    BOXSIZE, KVALS, RGRID, NRAND, MASS_CUT, TOP_FRACTION, MIN_AGN,
    RANDOM_SEED, SNAP, SIM_PATH, OUTPUT_DIR,
)
from .data_io import find_snapshots, sim_id_from_path, read_bh_catalog
from .selection import select_luminous_agn
from .knn_cdf import knn_summary


def _process_snapshot(args):
    snapshot_path, mass_cut, top_fraction, min_agn, boxsize, kvals, rgrid, random_pos = args

    try:
        catalog = read_bh_catalog(snapshot_path)
        if catalog is None:
            return None

        mask = select_luminous_agn(
            catalog["bh_mass"], catalog["bh_mdot"], mass_cut, top_fraction
        )

        pos = catalog["pos"][mask]
        n_agn = len(pos)
        if n_agn < max(min_agn, max(kvals) + 1):
            return None

        summary = knn_summary(pos, boxsize, kvals, rgrid, random_pos)
        sim_id = sim_id_from_path(snapshot_path)

        return sim_id, summary, n_agn

    except Exception as exc:  # noqa: BLE001 - one bad snapshot shouldn't kill the run
        print(f"\nFAILED: {snapshot_path}\n  {type(exc).__name__}: {exc}")
        return None


def run_suite(
    snap=SNAP,
    mass_cut=MASS_CUT,
    top_fraction=TOP_FRACTION,
    min_agn=MIN_AGN,
    nrand=NRAND,
    kvals=KVALS,
    boxsize=BOXSIZE,
    rgrid=RGRID,
    nproc=None,
    sim_path=SIM_PATH,
    output_dir=OUTPUT_DIR,
    random_seed=RANDOM_SEED,
):
    """
    Compute the AGN kNN-CDF for every LH simulation at `snap` and save
    the result to `<output_dir>/agn_knn_snap<snap>_M<mass_cut>_top<pct>.npz`.

    Returns the same dict that's saved, for direct use without a
    round-trip through disk.
    """
    if nproc is None:
        nproc = min(16, os.cpu_count() or 1)

    rng = np.random.default_rng(random_seed)
    random_pos = rng.uniform(0, boxsize, size=(nrand, 3))

    files = find_snapshots(sim_path, snap)
    print(f"Found {len(files)} snapshots")

    worker_args = [
        (fname, mass_cut, top_fraction, min_agn, boxsize, kvals, rgrid, random_pos)
        for fname in files
    ]

    with Pool(nproc) as pool:
        results = list(
            tqdm(pool.imap(_process_snapshot, worker_args), total=len(worker_args))
        )

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r[0])  # deterministic sim_id order, every run

    sim_ids = np.array([r[0] for r in results])
    summaries = np.array([r[1] for r in results])
    n_agn = np.array([r[2] for r in results])

    os.makedirs(output_dir, exist_ok=True)
    pct = int(round(top_fraction * 100))
    outfile = (
        f"{output_dir}/agn_knn_snap{snap}_M{mass_cut:.0e}_top{pct}.npz"
    )

    np.savez(
        outfile,
        sim_ids=sim_ids,
        summaries=summaries,
        nbh=n_agn,
        rgrid=rgrid,
        kvals=np.array(kvals),
        snap=snap,
        mass_cut=mass_cut,
        top_fraction=top_fraction,
    )

    print(f"\nSaved: {outfile}")
    print(f"Successful runs = {len(results)}")
    print(f"Summary shape   = {summaries.shape}")
    print(f"Mean N_AGN      = {np.mean(n_agn):.1f}" if len(n_agn) else "Mean N_AGN = n/a")

    return {
        "sim_ids": sim_ids,
        "summaries": summaries,
        "nbh": n_agn,
        "rgrid": rgrid,
        "kvals": np.array(kvals),
    }
