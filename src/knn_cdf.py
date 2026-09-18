"""
kNN-CDF summary statistic (Banerjee & Abel 2021).

For a set of tracer positions in a periodic box, and k in KVALS, the
kNN-CDF is the empirical CDF of "distance from a random point to its
k-th nearest tracer", evaluated on a fixed radial grid so that summaries
from different simulations are directly comparable bin-by-bin.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


def cdf_on_grid(distances, rgrid):
    """
    Empirical CDF of `distances`, linearly interpolated onto `rgrid`.

    Below the smallest observed distance the CDF is 0; above the largest
    it is 1 (`fill_value=(0, 1)`), which is the correct extrapolation for
    a CDF.
    """
    rs = np.sort(distances)
    fs = np.arange(1, len(rs) + 1) / len(rs)
    return interp1d(rs, fs, bounds_error=False, fill_value=(0, 1))(rgrid)


def knn_summary(pos, boxsize, kvals, rgrid, random_pos):
    """
    kNN-CDF summary for one tracer set.

    Parameters
    ----------
    pos        : (N, 3) tracer positions, in the same units as boxsize
    boxsize    : periodic box size
    kvals      : sequence of k values, e.g. (1, 2, 4)
    rgrid      : radial grid the CDFs are read on
    random_pos : (n_rand, 3) query points (shared across sims so the
                 Poisson-sampling noise is identical, not just
                 independent, for every simulation)

    Returns
    -------
    summary : 1-D ndarray, length len(kvals) * len(rgrid), the per-k
              CDFs concatenated in k order (k-major layout):
                  flat[i] = summary block for kvals[i // len(rgrid)]
                            at rgrid[i % len(rgrid)]
    """
    tree = cKDTree(pos, boxsize=boxsize)
    dist, _ = tree.query(random_pos, k=list(kvals))

    # cKDTree.query collapses the k-axis when kvals is a single int; kvals
    # is always a sequence here, so dist is always (n_rand, len(kvals)).
    blocks = [cdf_on_grid(dist[:, i], rgrid) for i in range(len(kvals))]
    return np.concatenate(blocks)
