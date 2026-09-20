"""
run_suite() end-to-end, with I/O replaced by fakes -- no real HDF5 snapshots
or CAMELS data needed. Exercises exactly the new plumbing (dir_prefix,
numeric_id, tracer, output filename tagging) that the 1P/mass-control work
added, using nproc=1 so it stays fast and deterministic.
"""

import numpy as np
import pytest

import src.pipeline as pipeline


def _fake_catalog(n, seed, mass_range=(1e6, 1e9), mdot_range=(1e-6, 1e-1)):
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.uniform(0, 25.0, size=(n, 3)),
        "bh_mass": 10 ** rng.uniform(*np.log10(mass_range), size=n),
        "bh_mdot": 10 ** rng.uniform(*np.log10(mdot_range), size=n),
    }


@pytest.fixture
def fake_lh_snapshots(monkeypatch):
    paths = [f"/fake/LH/LH_{i}/snapshot_050.hdf5" for i in range(6)]

    def fake_find_snapshots(sim_path, snap, prefix="LH_"):
        return [p for p in paths if prefix in p]

    def fake_read_bh_catalog(path):
        sim_id = int(path.split("LH_")[1].split("/")[0])
        return _fake_catalog(n=100 + 10 * sim_id, seed=sim_id)

    monkeypatch.setattr(pipeline, "find_snapshots", fake_find_snapshots)
    monkeypatch.setattr(pipeline, "read_bh_catalog", fake_read_bh_catalog)
    return paths


@pytest.fixture
def fake_1p_snapshots(monkeypatch):
    labels = ["p1_0", "p1_1", "p1_2", "p2_0", "p2_1"]
    paths = [f"/fake/1P/1P_{lab}/snapshot_050.hdf5" for lab in labels]

    def fake_find_snapshots(sim_path, snap, prefix="LH_"):
        return [p for p in paths if prefix in p]

    def fake_read_bh_catalog(path):
        label = path.split("1P_")[1].split("/")[0]
        seed = abs(hash(label)) % (2**31)
        return _fake_catalog(n=150, seed=seed)

    monkeypatch.setattr(pipeline, "find_snapshots", fake_find_snapshots)
    monkeypatch.setattr(pipeline, "read_bh_catalog", fake_read_bh_catalog)
    return paths, labels


@pytest.fixture
def fake_cv_snapshots(monkeypatch):
    # CV uses the same simple integer-id convention as LH (CV_<n>), just a
    # different Set name -- no special label parsing needed, unlike 1P.
    paths = [f"/fake/CV/CV_{i}/snapshot_050.hdf5" for i in range(27)]

    def fake_find_snapshots(sim_path, snap, prefix="LH_"):
        return [p for p in paths if prefix in p]

    def fake_read_bh_catalog(path):
        sim_id = int(path.split("CV_")[1].split("/")[0])
        return _fake_catalog(n=200, seed=1000 + sim_id)  # same fiducial "recipe", different seed

    monkeypatch.setattr(pipeline, "find_snapshots", fake_find_snapshots)
    monkeypatch.setattr(pipeline, "read_bh_catalog", fake_read_bh_catalog)
    return paths


def test_run_suite_lh_default_is_backward_compatible(fake_lh_snapshots, tmp_path):
    result = pipeline.run_suite(
        n_target=50, sim_path="unused", output_dir=str(tmp_path), nproc=1,
    )

    assert result["sim_ids"].dtype.kind in "iu"  # integer, not object/string
    assert list(result["sim_ids"]) == sorted(result["sim_ids"])
    assert np.all(result["nbh"] == 50)  # fixed-N held exactly

    outfiles = list(tmp_path.glob("agn_knn_*_n50.npz"))
    assert len(outfiles) == 1, f"expected the unchanged 'agn_knn_..._n50.npz' name, got {list(tmp_path.iterdir())}"


def test_run_suite_mass_tracer_uses_distinct_filename(fake_lh_snapshots, tmp_path):
    pipeline.run_suite(
        n_target=50, tracer="mass", sim_path="unused", output_dir=str(tmp_path), nproc=1,
    )
    assert list(tmp_path.glob("bhmass_knn_*_n50.npz")), list(tmp_path.iterdir())
    assert not list(tmp_path.glob("agn_knn_*.npz"))


def test_run_suite_mass_and_luminosity_can_select_differently(fake_lh_snapshots, tmp_path):
    lum = pipeline.run_suite(
        n_target=50, tracer="luminosity", sim_path="unused",
        output_dir=str(tmp_path), nproc=1,
    )
    mass = pipeline.run_suite(
        n_target=50, tracer="mass", sim_path="unused",
        output_dir=str(tmp_path), nproc=1,
    )
    assert list(lum["sim_ids"]) == list(mass["sim_ids"])  # same sims retained
    assert not np.array_equal(lum["summaries"], mass["summaries"])  # different tracers picked


def test_run_suite_1p_produces_string_sim_ids_and_distinct_filename(fake_1p_snapshots, tmp_path):
    result = pipeline.run_suite(
        n_target=50, sim_path="unused", output_dir=str(tmp_path), nproc=1,
        dir_prefix="1P_", numeric_id=False,
    )

    assert result["sim_ids"].dtype == object
    assert set(result["sim_ids"]) == {"p1_0", "p1_1", "p1_2", "p2_0", "p2_1"}

    outfiles = list(tmp_path.glob("1p_knn_*_n50.npz"))
    assert len(outfiles) == 1, list(tmp_path.iterdir())

    # round-trips through np.savez/np.load without corrupting the string labels
    loaded = np.load(outfiles[0], allow_pickle=True)
    assert set(loaded["sim_ids"]) == set(result["sim_ids"])


def test_run_suite_cv_uses_integer_ids_and_distinct_filename(fake_cv_snapshots, tmp_path):
    result = pipeline.run_suite(
        n_target=150, sim_path="unused", output_dir=str(tmp_path), nproc=1,
        dir_prefix="CV_",
    )

    assert result["sim_ids"].dtype.kind in "iu"  # int, same convention as LH
    assert set(result["sim_ids"]) == set(range(27))
    assert np.all(result["nbh"] == 150)

    outfiles = list(tmp_path.glob("cv_knn_*_n150.npz"))
    assert len(outfiles) == 1, list(tmp_path.iterdir())
    assert not list(tmp_path.glob("agn_knn_*.npz"))

    # round-trips through np.savez/np.load without corrupting the ids
    loaded = np.load(outfiles[0], allow_pickle=True)
    assert set(loaded["sim_ids"]) == set(result["sim_ids"])


def _fake_galaxy_catalog(n, seed, mass_range=(1e7, 1e12)):
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.uniform(0, 25.0, size=(n, 3)),
        "stellar_mass": 10 ** rng.uniform(*np.log10(mass_range), size=n),
        "flag": np.ones(n, dtype=bool),
    }


@pytest.fixture
def fake_lh_galaxy_catalogs(monkeypatch):
    paths = [f"/fake/Groups/LH/LH_{i}/groups_050.hdf5" for i in range(6)]

    def fake_find_group_catalogs(groups_path, snap, prefix="LH_"):
        return [p for p in paths if prefix in p]

    def fake_read_galaxy_catalog(path):
        sim_id = int(path.split("LH_")[1].split("/")[0])
        return _fake_galaxy_catalog(n=200 + 20 * sim_id, seed=sim_id)

    monkeypatch.setattr(pipeline, "find_group_catalogs", fake_find_group_catalogs)
    monkeypatch.setattr(pipeline, "read_galaxy_catalog", fake_read_galaxy_catalog)
    return paths


def test_run_galaxy_suite_requires_n_target(fake_lh_galaxy_catalogs, tmp_path):
    with pytest.raises(ValueError):
        pipeline.run_galaxy_suite(output_dir=str(tmp_path), nproc=1)


def test_run_galaxy_suite_uses_integer_ids_and_distinct_filename(fake_lh_galaxy_catalogs, tmp_path):
    result = pipeline.run_galaxy_suite(
        n_target=100, output_dir=str(tmp_path), nproc=1,
    )

    assert result["sim_ids"].dtype.kind in "iu"
    assert list(result["sim_ids"]) == sorted(result["sim_ids"])
    assert np.all(result["nbh"] == 100)

    outfiles = list(tmp_path.glob("galaxy_knn_*_n100.npz"))
    assert len(outfiles) == 1, list(tmp_path.iterdir())
    assert not list(tmp_path.glob("agn_knn_*.npz"))

    loaded = np.load(outfiles[0], allow_pickle=True)
    assert set(loaded["sim_ids"]) == set(result["sim_ids"])


def test_run_galaxy_suite_drops_sims_below_n_target(fake_lh_galaxy_catalogs, tmp_path):
    # fake catalogs have 200 + 20*sim_id eligible-ish galaxies; sim_id=0
    # has the fewest (~200), so a high n_target should drop it
    result = pipeline.run_galaxy_suite(
        n_target=205, output_dir=str(tmp_path), nproc=1,
    )
    assert 0 not in result["sim_ids"]
    assert np.all(result["nbh"] == 205)


def test_run_galaxy_suite_cv_gets_its_own_filename_not_overwriting_lh(monkeypatch, tmp_path):
    # Regression: run_galaxy_suite used to always save to
    # "galaxy_knn_..." regardless of dir_prefix, so a CV run at the same
    # N as an already-saved LH run would silently clobber it. One fake
    # find/read pair covers both prefixes so both calls share it, rather
    # than composing two fixtures that would monkeypatch over each other.
    lh_paths = [f"/fake/LH/LH_{i}/groups_050.hdf5" for i in range(6)]
    cv_paths = [f"/fake/CV/CV_{i}/groups_050.hdf5" for i in range(6)]
    all_paths = lh_paths + cv_paths

    def fake_find_group_catalogs(groups_path, snap, prefix="LH_"):
        return [p for p in all_paths if prefix in p]

    def fake_read_galaxy_catalog(path):
        label = path.split("/")[-2]  # "LH_3" or "CV_3"
        sim_id = int(label.split("_")[1])
        return _fake_galaxy_catalog(n=200, seed=hash(label) % 10_000)

    monkeypatch.setattr(pipeline, "find_group_catalogs", fake_find_group_catalogs)
    monkeypatch.setattr(pipeline, "read_galaxy_catalog", fake_read_galaxy_catalog)

    pipeline.run_galaxy_suite(n_target=100, output_dir=str(tmp_path), nproc=1)  # LH
    lh_files = set(tmp_path.glob("galaxy_knn_*_n100.npz"))
    assert len(lh_files) == 1

    pipeline.run_galaxy_suite(
        n_target=100, output_dir=str(tmp_path), nproc=1,
        groups_path="unused", dir_prefix="CV_",
    )  # CV, deliberately same N

    assert lh_files == set(tmp_path.glob("galaxy_knn_*_n100.npz")), "LH file was overwritten"
    cv_files = list(tmp_path.glob("galaxy_cv_knn_*_n100.npz"))
    assert len(cv_files) == 1, list(tmp_path.iterdir())


def test_run_suite_drops_sims_below_n_target(fake_lh_snapshots, tmp_path):
    # fake catalogs have 100 + 10*sim_id eligible-ish BHs; sim_id=0 has the
    # fewest (~100), so a high n_target should drop it while keeping others
    result = pipeline.run_suite(
        n_target=105, sim_path="unused", output_dir=str(tmp_path), nproc=1,
    )
    assert 0 not in result["sim_ids"]
    assert np.all(result["nbh"] == 105)
