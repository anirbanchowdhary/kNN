"""
Reading CAMELS-IllustrisTNG LH snapshots.

Layout on disk (fixed by the simulation suite, not something we choose):

    <sim_path>/LH_<id>/snapshot_<snap:03d>.hdf5

Each snapshot's "PartType5" group holds every black hole in the box.
"""

import os

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filters used by CAMELS)
import numpy as np


def find_snapshots(sim_path, snap, prefix="LH_"):
    """
    Return the sorted list of snapshot file paths, one per simulation
    directory under `sim_path` whose name starts with `prefix`, for a given
    snapshot number.

    `prefix` generalizes this beyond the LH set: CAMELS's 1P set uses
    `1P_p<param>_<step>` directories at the same nesting level, sitting
    alongside `LH_<id>`.
    """
    snap_name = f"snapshot_{snap:03d}.hdf5"
    with os.scandir(sim_path) as top:
        return sorted(
            os.path.join(entry.path, snap_name)
            for entry in top
            if entry.is_dir(follow_symlinks=False) and entry.name.startswith(prefix)
        )


def parse_dir_label(path, prefix):
    """
    Extract the simulation-directory label following `prefix` from a
    snapshot path, e.g. `parse_dir_label(".../LH_42/snapshot_050.hdf5", "LH_")
    == "42"`, or `parse_dir_label(".../1P_p1_3/...", "1P_") == "p1_3"`.
    """
    return path.split(prefix)[1].split("/")[0]


def sim_id_from_path(path):
    """Extract the integer LH simulation id from a snapshot path."""
    return int(parse_dir_label(path, "LH_"))


def read_bh_catalog(snapshot_path):
    """
    Read the black-hole (PartType5) catalog from one snapshot.

    Returns
    -------
    dict with keys "pos" (N,3) in cMpc/h, "bh_mass" (N,) in Msun,
    "bh_mdot" (N,) in code units, or None if the snapshot has no black
    holes (e.g. very early snapshots).
    """
    with h5py.File(snapshot_path, "r") as f:
        if "PartType5" not in f:
            return None

        h = f["Header"].attrs["HubbleParam"]

        pos = f["PartType5"]["Coordinates"][:].astype(np.float64) / 1e3
        bh_mass = f["PartType5"]["BH_Mass"][:].astype(np.float64) * 1e10 / h
        bh_mdot = f["PartType5"]["BH_Mdot"][:].astype(np.float64)

    return {"pos": pos, "bh_mass": bh_mass, "bh_mdot": bh_mdot}
