import numpy as np

from src.data_io import parse_dir_label, sim_id_from_path, wrap_periodic


def test_parse_dir_label_lh():
    path = "/data/Sims/IllustrisTNG/L25n256/LH/LH_42/snapshot_050.hdf5"
    assert parse_dir_label(path, "LH_") == "42"


def test_parse_dir_label_1p():
    path = "/data/Sims/IllustrisTNG/L25n256/1P/1P_p1_3/snapshot_050.hdf5"
    assert parse_dir_label(path, "1P_") == "p1_3"


def test_sim_id_from_path_matches_parse_dir_label():
    path = "/data/Sims/IllustrisTNG/L25n256/LH/LH_7/snapshot_050.hdf5"
    assert sim_id_from_path(path) == 7
    assert isinstance(sim_id_from_path(path), int)


def test_parse_dir_label_uses_first_prefix_occurrence():
    # split() matches the FIRST occurrence of `prefix` in the path, so if
    # sim_path's own directory tree happens to contain the prefix string
    # earlier, the label comes out wrong. find_snapshots always builds
    # paths as <sim_path>/<prefix><id>/..., so this only bites if sim_path
    # itself contains the prefix -- documented here rather than silently
    # relied on.
    path = "/data/LH_suite/Sims/LH_123/snapshot_090.hdf5"
    assert parse_dir_label(path, "LH_") == "suite"


def test_wrap_periodic_leaves_interior_points_unchanged():
    pos = np.array([[0.0, 12.5, 24.9999], [1.0, 1.0, 1.0]])
    assert np.allclose(wrap_periodic(pos, boxsize=25.0), pos)


def test_wrap_periodic_wraps_negative_and_overflow_values():
    # the real bug this guards against: SubhaloPos landing marginally
    # outside [0, boxsize) -- exactly at the edge, or a hair negative --
    # which scipy's boxsize-aware cKDTree rejects outright
    pos = np.array([[-1e-10, 25.0, 26.0], [24.999999, -0.5, 25.0]])
    wrapped = wrap_periodic(pos, boxsize=25.0)
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 25.0)
    assert np.isclose(wrapped[0, 0], 25.0, atol=1e-6)  # -1e-10 wraps to ~boxsize, not 0
    assert np.isclose(wrapped[0, 1], 0.0)
    assert np.isclose(wrapped[0, 2], 1.0)
    assert np.isclose(wrapped[1, 1], 24.5)
