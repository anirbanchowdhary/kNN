import numpy as np
import pytest

from src.complementarity import bin_correlation, align_common_sims


def test_bin_correlation_is_one_for_identical_inputs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 12))
    corr = bin_correlation(a, a.copy())
    assert np.allclose(corr, 1.0)


def test_bin_correlation_is_near_zero_for_independent_inputs():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 8))
    b = rng.normal(size=(500, 8))
    corr = bin_correlation(a, b)
    assert np.all(np.abs(corr) < 0.2)  # loose: finite-n noise, not exactly 0


def test_bin_correlation_is_minus_one_for_perfectly_anticorrelated_inputs():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(30, 5))
    corr = bin_correlation(a, -a)
    assert np.allclose(corr, -1.0)


def test_bin_correlation_shape_is_bin_count():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20, 15))
    b = rng.normal(size=(20, 15))
    corr = bin_correlation(a, b)
    assert corr.shape == (15,)


def test_bin_correlation_rejects_mismatched_shapes():
    a = np.zeros((10, 5))
    b = np.zeros((10, 6))
    with pytest.raises(ValueError):
        bin_correlation(a, b)


def test_bin_correlation_rejects_too_few_sims():
    a = np.zeros((2, 5))
    b = np.zeros((2, 5))
    with pytest.raises(ValueError):
        bin_correlation(a, b)


def test_align_common_sims_intersects_and_reorders_consistently():
    sim_ids_a = np.array([3, 1, 5, 2])
    residuals_a = np.array([[30], [10], [50], [20]])
    sim_ids_b = np.array([2, 5, 9])
    residuals_b = np.array([[200], [500], [900]])

    common, aligned_a, aligned_b = align_common_sims(sim_ids_a, residuals_a, sim_ids_b, residuals_b)

    assert list(common) == [2, 5]
    # row i of aligned_a/aligned_b both correspond to common[i]
    assert aligned_a.flatten().tolist() == [20, 50]
    assert aligned_b.flatten().tolist() == [200, 500]


def test_align_common_sims_raises_when_disjoint():
    with pytest.raises(ValueError):
        align_common_sims(
            np.array([1, 2]), np.zeros((2, 3)),
            np.array([3, 4]), np.zeros((2, 3)),
        )
