import numpy as np
import pytest

from src.cv import per_bin_std, scalar_noise_floor, signal_to_noise, MIN_REALIZATIONS


def _synthetic_cv_summaries(n_sims, n_k, n_r, bin_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n_total = n_k * n_r
    return rng.normal(0, bin_std, size=(n_sims, n_total))


def test_per_bin_std_matches_known_input_scale():
    n_k, n_r = 3, 40
    summaries = _synthetic_cv_summaries(30, n_k, n_r, bin_std=0.05)

    std = per_bin_std(summaries, n_k, n_r)

    assert std.shape == (n_k, n_r)
    assert np.all(std > 0)
    # with 30 draws from N(0, 0.05), the per-bin sample std should land
    # close to 0.05 -- not exact, but well within a generous band
    assert np.median(std) == pytest.approx(0.05, rel=0.3)


def test_scalar_noise_floor_is_rms_of_per_bin_std():
    n_k, n_r = 2, 20
    summaries = _synthetic_cv_summaries(20, n_k, n_r)

    scalar = scalar_noise_floor(summaries)
    std = per_bin_std(summaries, n_k, n_r)

    assert scalar == pytest.approx(np.sqrt(np.mean(std**2)))


def test_per_bin_std_raises_below_min_realizations():
    n_k, n_r = 2, 10
    summaries = _synthetic_cv_summaries(MIN_REALIZATIONS - 1, n_k, n_r)

    with pytest.raises(ValueError):
        per_bin_std(summaries, n_k, n_r)


def test_scalar_noise_floor_raises_below_min_realizations():
    summaries = _synthetic_cv_summaries(MIN_REALIZATIONS - 1, 2, 10)
    with pytest.raises(ValueError):
        scalar_noise_floor(summaries)


def test_per_bin_std_accepts_exactly_min_realizations():
    n_k, n_r = 2, 10
    summaries = _synthetic_cv_summaries(MIN_REALIZATIONS, n_k, n_r)
    std = per_bin_std(summaries, n_k, n_r)  # must not raise
    assert std.shape == (n_k, n_r)


def test_signal_to_noise_is_elementwise_ratio():
    response = np.array([[0.1, -0.2], [0.3, 0.4]])
    cv_std = np.array([[0.1, 0.1], [0.1, 0.2]])

    ratio = signal_to_noise(response, cv_std)

    assert ratio.shape == response.shape
    assert np.allclose(ratio, [[1.0, 2.0], [3.0, 2.0]])
    assert np.all(ratio >= 0)  # abs() applied -- sign of response discarded


def test_signal_to_noise_scalar_case():
    assert signal_to_noise(-0.06, 0.02) == pytest.approx(3.0)
