import numpy as np

from src.knn_cdf import cdf_on_grid, knn_summary


def test_cdf_on_grid_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    distances = rng.uniform(0, 10, size=500)
    rgrid = np.linspace(0, 15, 30)

    cdf = cdf_on_grid(distances, rgrid)

    assert np.all(cdf >= 0) and np.all(cdf <= 1)
    assert np.all(np.diff(cdf) >= -1e-12)  # monotonically non-decreasing
    assert cdf[0] == 0  # below the smallest observed distance
    assert cdf[-1] == 1  # above the largest observed distance


def test_knn_summary_shape_and_layout():
    rng = np.random.default_rng(1)
    boxsize = 25.0
    pos = rng.uniform(0, boxsize, size=(200, 3))
    random_pos = rng.uniform(0, boxsize, size=(50, 3))
    kvals = (1, 2, 4)
    rgrid = np.logspace(-1.5, 1.2, 20)

    summary = knn_summary(pos, boxsize, kvals, rgrid, random_pos)

    assert summary.shape == (len(kvals) * len(rgrid),)

    # k-major layout: block i is the CDF for kvals[i]
    blocks = summary.reshape(len(kvals), len(rgrid))
    for block in blocks:
        assert np.all(np.diff(block) >= -1e-12)


def test_larger_k_has_larger_typical_distance():
    # The k=4 nearest-neighbor distance CDF should sit at or to the right
    # of the k=1 CDF (same location climbs its CDF later => equal-or-lower
    # value at a fixed r), since a 4th neighbor is never closer than a 1st.
    rng = np.random.default_rng(2)
    boxsize = 25.0
    pos = rng.uniform(0, boxsize, size=(300, 3))
    random_pos = rng.uniform(0, boxsize, size=(200, 3))
    kvals = (1, 4)
    rgrid = np.logspace(-1.5, 1.2, 20)

    summary = knn_summary(pos, boxsize, kvals, rgrid, random_pos)
    cdf_k1, cdf_k4 = summary.reshape(len(kvals), len(rgrid))

    assert np.all(cdf_k1 >= cdf_k4 - 1e-9)
