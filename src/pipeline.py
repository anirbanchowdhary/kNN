"""
End-to-end run: every simulation's snapshot -> one kNN-CDF summary.

Two selection modes, chosen via `n_target`:

- **fixed-N** (`n_target` set): exactly the N most-extreme BHs per
  simulation by `tracer` ("luminosity" or "mass"). Tracer number density is
  then identical across the suite by construction, so the kNN-CDF's strong,
  nonlinear dependence on number density is held fixed rather than regressed
  out afterwards. Preferred -- `abundance.remove_abundance` is a
  linear-in-log10(n) approximation, which cannot fully remove a nonlinear
  dependence, and any parameter correlated with n (Omega_m, strongly)
  inherits the leftover.
- **top-fraction** (`n_target=None`, default): top `top_fraction` by
  luminosity, so N varies per simulation and the abundance trend must be
  removed downstream. `tracer` is ignored in this mode (luminosity only).

`dir_prefix`/`numeric_id` generalize this beyond the LH set: pass
`dir_prefix="1P_"`, `numeric_id=False` to run over CAMELS's 1P set instead,
whose directories are named `1P_p<param>_<step>` rather than `LH_<int>` and
have no single integer id.

Parallelized across simulations, since each snapshot is independent.
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    BOXSIZE, KVALS, RGRID, NRAND, MASS_CUT, TOP_FRACTION, MIN_AGN,
    RANDOM_SEED, SNAP, SIM_PATH, OUTPUT_DIR,
)
from .data_io import find_snapshots, parse_dir_label, read_bh_catalog
from .selection import (
    select_luminous_agn, select_brightest_n, select_most_massive_n, count_eligible,
)
from .knn_cdf import knn_summary


def _default_nproc(nproc):
    return min(16, os.cpu_count() or 1) if nproc is None else nproc


def _select(catalog, mass_cut, top_fraction, n_target, tracer, kvals):
    if n_target is not None:
        if tracer == "mass":
            return select_most_massive_n(catalog["bh_mass"], mass_cut, n_target)
        return select_brightest_n(catalog["bh_mass"], catalog["bh_mdot"], mass_cut, n_target)
    return select_luminous_agn(catalog["bh_mass"], catalog["bh_mdot"], mass_cut, top_fraction)


# ----------------------------------------------------------------------
# Counting eligible AGN (to choose n_target before committing to a run)
# ----------------------------------------------------------------------

def _count_snapshot(args):
    snapshot_path, mass_cut, dir_prefix = args
    try:
        catalog = read_bh_catalog(snapshot_path)
        if catalog is None:
            return None
        n = count_eligible(catalog["bh_mass"], catalog["bh_mdot"], mass_cut)
        return parse_dir_label(snapshot_path, dir_prefix), n
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {snapshot_path}\n  {type(exc).__name__}: {exc}")
        return None


def survey_eligible_counts(
    snap=SNAP, mass_cut=MASS_CUT, sim_path=SIM_PATH, dir_prefix="LH_", nproc=None,
):
    """
    How many BHs pass the mass floor (with usable luminosities) in each
    simulation, without building any kd-trees.

    Use this to pick `n_target` for a fixed-N run: a low percentile of this
    distribution keeps the density high while dropping few simulations.

    Returns a DataFrame with columns [sim_id, n_eligible]. `sim_id` is the
    raw directory-label string (e.g. "42" for LH, "p1_3" for 1P).
    """
    files = find_snapshots(sim_path, snap, prefix=dir_prefix)
    print(f"Found {len(files)} snapshots")

    with Pool(_default_nproc(nproc)) as pool:
        results = list(
            tqdm(
                pool.imap(_count_snapshot, [(f, mass_cut, dir_prefix) for f in files]),
                total=len(files),
            )
        )

    results = sorted(r for r in results if r is not None)
    return pd.DataFrame(results, columns=["sim_id", "n_eligible"])


# ----------------------------------------------------------------------
# kNN-CDF summaries
# ----------------------------------------------------------------------

def _process_snapshot(args):
    (
        snapshot_path, mass_cut, top_fraction, n_target, tracer, min_agn,
        boxsize, kvals, rgrid, random_pos, dir_prefix,
    ) = args

    try:
        catalog = read_bh_catalog(snapshot_path)
        if catalog is None:
            return None

        mask = _select(catalog, mass_cut, top_fraction, n_target, tracer, kvals)

        pos = catalog["pos"][mask]
        n_agn = len(pos)
        if n_agn < max(min_agn, max(kvals) + 1):
            return None

        summary = knn_summary(pos, boxsize, kvals, rgrid, random_pos)
        label = parse_dir_label(snapshot_path, dir_prefix)

        return label, summary, n_agn

    except Exception as exc:  # noqa: BLE001 - one bad snapshot shouldn't kill the run
        print(f"\nFAILED: {snapshot_path}\n  {type(exc).__name__}: {exc}")
        return None


def run_suite(
    snap=SNAP,
    mass_cut=MASS_CUT,
    top_fraction=TOP_FRACTION,
    n_target=None,
    tracer="luminosity",
    min_agn=MIN_AGN,
    nrand=NRAND,
    kvals=KVALS,
    boxsize=BOXSIZE,
    rgrid=RGRID,
    nproc=None,
    sim_path=SIM_PATH,
    output_dir=OUTPUT_DIR,
    random_seed=RANDOM_SEED,
    dir_prefix="LH_",
    numeric_id=True,
):
    """
    Compute the kNN-CDF for every simulation at `snap` and save to
    `<output_dir>/<tag>_knn_snap<snap>_M<mass_cut>_{n<N>|top<pct>}.npz`.

    Pass `n_target` for a fixed-number-density run (recommended); leave it
    None to select the top `top_fraction` by luminosity instead. `tracer`
    ("luminosity" or "mass") only applies in fixed-N mode.

    `dir_prefix`/`numeric_id` select the simulation set: defaults are the LH
    set (`LH_<int>` directories, integer sim_ids). Pass `dir_prefix="1P_",
    numeric_id=False` for the 1P set, whose directories are `1P_p<N>_<M>`
    and have no single integer id -- `sim_ids` is then a string array of
    labels like "p1_3" instead.

    Returns the same dict that's saved, for use without a disk round-trip.
    """
    rng = np.random.default_rng(random_seed)
    random_pos = rng.uniform(0, boxsize, size=(nrand, 3))

    files = find_snapshots(sim_path, snap, prefix=dir_prefix)
    print(f"Found {len(files)} snapshots")

    worker_args = [
        (f, mass_cut, top_fraction, n_target, tracer, min_agn, boxsize, kvals,
         rgrid, random_pos, dir_prefix)
        for f in files
    ]

    with Pool(_default_nproc(nproc)) as pool:
        results = list(
            tqdm(pool.imap(_process_snapshot, worker_args), total=len(worker_args))
        )

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r[0])  # deterministic sim_id order, every run

    if numeric_id:
        sim_ids = np.array([int(r[0]) for r in results])
    else:
        sim_ids = np.array([r[0] for r in results], dtype=object)
    summaries = np.array([r[1] for r in results])
    n_agn = np.array([r[2] for r in results])

    os.makedirs(output_dir, exist_ok=True)
    tag = f"n{n_target}" if n_target is not None else f"top{int(round(top_fraction * 100))}"

    if dir_prefix == "LH_":
        # Unchanged for the default (tracer="luminosity") case, so this
        # doesn't rename files already on disk from earlier runs; "bhmass"
        # for the mass-selected control keeps it distinguishable in the same
        # output_dir without colliding with or being mistaken for the AGN run.
        set_tag = "agn" if tracer == "luminosity" else "bhmass"
    else:
        set_tag = dir_prefix.strip("_").lower()  # e.g. "1p"
    outfile = f"{output_dir}/{set_tag}_knn_snap{snap}_M{mass_cut:.0e}_{tag}.npz"

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
        n_target=-1 if n_target is None else n_target,
        tracer=tracer,
    )

    print(f"\nSaved: {outfile}")
    print(f"Successful runs = {len(results)} / {len(files)}")
    print(f"Summary shape   = {summaries.shape}")
    if len(n_agn):
        print(f"N_AGN: mean {np.mean(n_agn):.1f}, min {n_agn.min()}, max {n_agn.max()}")

    return {
        "sim_ids": sim_ids,
        "summaries": summaries,
        "nbh": n_agn,
        "rgrid": rgrid,
        "kvals": np.array(kvals),
    }
