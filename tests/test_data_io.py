from src.data_io import parse_dir_label, sim_id_from_path


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
