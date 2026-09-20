"""
CV set support: fiducial cosmology/astrophysics held fixed, only the
initial-condition random seed varies across realizations. This isolates
cosmic variance -- how much the kNN-CDF scatters between exact physical
replicas of the same universe, from nothing but chance -- as a noise
floor to compare a real response against, more directly than a
permutation test can at the small n a 1P sweep has (see
`onep.monotonic_trend`'s docstring) or than a bootstrap CI over 1000 LH
simulations already does on its own.
"""

import numpy as np

MIN_REALIZATIONS = 5  # below this, the std itself is too noisy to trust


def _check_enough_realizations(n_sims):
    if n_sims < MIN_REALIZATIONS:
        raise ValueError(
            f"only {n_sims} CV realizations given, need at least "
            f"{MIN_REALIZATIONS} for a trustworthy noise floor"
        )


def per_bin_std(summaries, n_k, n_r):
    """
    Per-bin standard deviation of the kNN-CDF across CV realizations,
    reshaped to (n_k, n_r) -- directly comparable to a response's per-bin
    diff (e.g. `sensitivity.scale_resolved_response`,
    `onep.mean_cdf_by_step`'s per-step mean minus the fiducial), since
    both live on the same summary statistic.

    Uses ddof=1 (sample std): with ~27 realizations this is a small but
    real correction, and the whole point here is not to be sloppy about
    what "noise" means.
    """
    _check_enough_realizations(summaries.shape[0])
    return summaries.std(axis=0, ddof=1).reshape(n_k, n_r)


def scalar_noise_floor(summaries):
    """RMS over bins of the per-bin std -- one number, comparable to R(p)."""
    _check_enough_realizations(summaries.shape[0])
    std = summaries.std(axis=0, ddof=1)
    return float(np.sqrt(np.mean(std**2)))


def signal_to_noise(response, cv_std):
    """
    Elementwise |response| / cv_std -- how many multiples of the pure
    seed-to-seed scatter a response's magnitude represents, at each bin.
    `response` and `cv_std` must be the same shape (both (n_k, n_r), or
    both scalars for the RMS-collapsed versions).

    This is a ratio, not a p-value: it doesn't carry a rigorous
    probabilistic guarantee the way the LH bootstrap/permutation
    machinery or an exact permutation test does, since it doesn't account
    for how `cv_std` itself was estimated from a finite (~27) sample.
    Read it as "how many noise-widths is this", not "the probability this
    is due to chance".
    """
    return np.abs(response) / cv_std
