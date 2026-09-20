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

from .config import BOXSIZE


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


def find_group_catalogs(groups_path, snap, prefix="LH_"):
    """
    Galaxy-tracer counterpart to `find_snapshots`: return the sorted list
    of SubFind group-catalog file paths, one per simulation directory
    under `groups_path` whose name starts with `prefix`, for a given
    snapshot number. Mirrors `find_snapshots` exactly -- for this
    account's data, `groups_path` is the same directory tree as
    `sim_path` (group catalogs sit alongside snapshots, confirmed by
    notebook 06's discovery cell), but the function takes its own path
    since CAMELS documents this as a logically separate data type and
    other accounts' data may actually be laid out that way.
    """
    cat_name = f"groups_{snap:03d}.hdf5"
    with os.scandir(groups_path) as top:
        return sorted(
            os.path.join(entry.path, cat_name)
            for entry in top
            if entry.is_dir(follow_symlinks=False) and entry.name.startswith(prefix)
        )


def wrap_periodic(pos, boxsize=BOXSIZE):
    """
    Wrap positions into [0, boxsize).

    `SubhaloPos` (a periodic-aware center-of-mass) can land marginally
    outside that half-open interval -- exactly at `boxsize`, or a hair
    negative -- for a subhalo whose particles straddle the box edge.
    `scipy.spatial.cKDTree`'s `boxsize=` mode rejects that outright
    ("Some/Negative input data are ... outside of the periodic box"),
    which silently dropped 2/1000 real LH simulations from the first
    galaxy-tracer run before this wrap was added. BH `Coordinates` have
    not shown the same issue in any real run so far, so this is applied
    only where it's actually needed.
    """
    return np.mod(pos, boxsize)


def read_galaxy_catalog(catalog_path):
    """
    Read the subhalo ("galaxy") catalog from one CAMELS group-catalog file.

    Assumes the standard AREPO/SubFind `Subhalo` group layout:
    `SubhaloPos` (comoving ckpc/h) and `SubhaloMassType`, whose 6 columns
    are [gas, DM, -, -, stars, BH] -- column 4 is stellar mass. Not
    independently verified against this account's actual data; a
    malformed/unexpected file raises with the offending path and the
    original exception rather than silently returning nonsense, and
    notebook 06's discovery cell prints what's actually inside the first
    file found before any of this is relied on downstream.

    Returns
    -------
    dict with keys "pos" (N,3) in cMpc/h, "stellar_mass" (N,) in Msun,
    "flag" (N,) bool (SubFind's not-spurious flag, all-True if the field
    is absent), or None if the catalog has no subhalos.
    """
    with h5py.File(catalog_path, "r") as f:
        if "Subhalo" not in f or f["Subhalo"]["SubhaloPos"].shape[0] == 0:
            return None

        h = f["Header"].attrs["HubbleParam"]

        pos = wrap_periodic(f["Subhalo"]["SubhaloPos"][:].astype(np.float64) / 1e3)
        stellar_mass = f["Subhalo"]["SubhaloMassType"][:, 4].astype(np.float64) * 1e10 / h
        if "SubhaloFlag" in f["Subhalo"]:
            flag = f["Subhalo"]["SubhaloFlag"][:].astype(bool)
        else:
            flag = np.ones(len(pos), dtype=bool)

    return {"pos": pos, "stellar_mass": stellar_mass, "flag": flag}
