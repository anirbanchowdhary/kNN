import numpy as np

from src.abundance import remove_abundance


def test_remove_abundance_strips_linear_trend():
    rng = np.random.default_rng(0)
    n_sims, n_bins = 200, 5

    nbh = rng.integers(50, 5000, size=n_sims)
    log_nbh = np.log10(nbh)

    # Each bin is a pure linear function of log10(nbh) plus small noise.
    slopes = rng.uniform(-1, 1, size=n_bins)
    intercepts = rng.uniform(-1, 1, size=n_bins)
    noise = rng.normal(0, 1e-3, size=(n_sims, n_bins))
    summaries = intercepts + slopes * log_nbh[:, None] + noise

    residuals = remove_abundance(summaries, nbh)

    # After removing the log10(nbh) trend, residuals should be ~noise-level
    # and uncorrelated with log10(nbh).
    assert np.all(np.abs(residuals).max(axis=0) < 0.05)
    for j in range(n_bins):
        corr = np.corrcoef(residuals[:, j], log_nbh)[0, 1]
        assert abs(corr) < 0.1


def test_remove_abundance_preserves_shape():
    rng = np.random.default_rng(1)
    summaries = rng.uniform(0, 1, size=(30, 8))
    nbh = rng.integers(10, 100, size=30)

    residuals = remove_abundance(summaries, nbh)

    assert residuals.shape == summaries.shape
