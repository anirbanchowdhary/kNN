import numpy as np
import pandas as pd
import pytest

from src.fisher import (
    linear_response,
    fisher_matrix,
    combine_fisher,
    marginalized_covariance,
    sub_covariance,
    confidence_ellipse_params,
    drop_uninformative_bins,
)


def test_linear_response_recovers_known_linear_coefficients():
    rng = np.random.default_rng(0)
    n_sims, n_bins = 500, 4
    theta = pd.DataFrame({
        "a": rng.uniform(-1, 1, n_sims),
        "b": rng.uniform(-1, 1, n_sims),
    })
    # bin j's true slope is (j+1, -(j+1)) for (a, b), plus tiny noise
    true_slopes = np.array([[j + 1, -(j + 1)] for j in range(n_bins)])
    residuals = (
        theta["a"].values[:, None] * true_slopes[:, 0][None, :]
        + theta["b"].values[:, None] * true_slopes[:, 1][None, :]
        + rng.normal(0, 1e-6, size=(n_sims, n_bins))
    )

    D = linear_response(residuals, theta, params=["a", "b"])

    assert D.shape == (2, n_bins)
    assert np.allclose(D[0], true_slopes[:, 0], atol=1e-3)
    assert np.allclose(D[1], true_slopes[:, 1], atol=1e-3)


def test_linear_response_rejects_misaligned_theta():
    residuals = np.zeros((10, 3))
    theta = pd.DataFrame({"a": np.zeros(9)})
    with pytest.raises(ValueError):
        linear_response(residuals, theta, params=["a"])


def test_drop_uninformative_bins_removes_exactly_zero_variance_bins():
    response = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    bin_variance = np.array([1.0, 0.0, 2.0, 0.0])

    kept_response, kept_variance, n_dropped = drop_uninformative_bins(response, bin_variance)

    assert n_dropped == 2
    assert np.array_equal(kept_response, np.array([[1.0, 3.0], [5.0, 7.0]]))
    assert np.array_equal(kept_variance, np.array([1.0, 2.0]))


def test_drop_uninformative_bins_keeps_everything_when_all_positive():
    response = np.ones((2, 3))
    bin_variance = np.array([1.0, 2.0, 3.0])
    kept_response, kept_variance, n_dropped = drop_uninformative_bins(response, bin_variance)
    assert n_dropped == 0
    assert kept_response.shape == response.shape
    assert np.array_equal(kept_variance, bin_variance)


def test_fisher_matrix_matches_direct_formula():
    response = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    bin_variance = np.array([1.0, 2.0, 4.0])

    F = fisher_matrix(response, bin_variance)

    expected = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            expected[i, j] = np.sum(response[i] * response[j] / bin_variance)

    assert np.allclose(F, expected)
    assert np.allclose(F, F.T)  # Fisher matrices are symmetric


def test_fisher_matrix_rejects_nonpositive_variance():
    response = np.ones((2, 3))
    bad_variance = np.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        fisher_matrix(response, bad_variance)


def test_fisher_matrix_rejects_shape_mismatch():
    response = np.ones((2, 3))
    bin_variance = np.ones(4)
    with pytest.raises(ValueError):
        fisher_matrix(response, bin_variance)


def test_combine_fisher_sums_matrices():
    F1 = np.array([[2.0, 0.0], [0.0, 3.0]])
    F2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    combined = combine_fisher(F1, F2)
    assert np.allclose(combined, F1 + F2)


def test_combine_fisher_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        combine_fisher(np.eye(2), np.eye(3))


def test_combine_fisher_requires_at_least_one_matrix():
    with pytest.raises(ValueError):
        combine_fisher()


def test_marginalized_covariance_inverts_diagonal_fisher():
    F = np.diag([4.0, 25.0])
    cov = marginalized_covariance(F)
    assert np.allclose(cov, np.diag([0.25, 0.04]))


def test_marginalized_covariance_raises_on_near_singular_matrix():
    F = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]])  # rank-deficient in practice
    with pytest.raises(ValueError):
        marginalized_covariance(F)


def test_sub_covariance_extracts_correct_block():
    cov = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 2.0, 0.3],
        [0.2, 0.3, 3.0],
    ])
    params = ["a", "b", "c"]
    sub = sub_covariance(cov, params, ["c", "a"])
    assert np.allclose(sub, [[3.0, 0.2], [0.2, 1.0]])


def test_marginalizing_then_slicing_differs_from_slicing_then_inverting():
    # The whole point of marginalized_covariance + sub_covariance: for a
    # parameter genuinely degenerate with a third one, inverting F's own
    # submatrix directly (profiling) gives a tighter, WRONG answer than
    # marginalizing over the full space first.
    F = np.array([
        [2.0, 1.8, 0.5],
        [1.8, 2.0, 0.3],
        [0.5, 0.3, 1.0],
    ])
    params = ["a", "b", "c"]

    correct = sub_covariance(marginalized_covariance(F), params, ["a", "b"])

    idx = [params.index(p) for p in ["a", "b"]]
    wrong_profiled = np.linalg.inv(F[np.ix_(idx, idx)])

    assert not np.allclose(correct, wrong_profiled)
    # marginalizing over the degenerate pair must widen the constraint
    assert correct[0, 0] > wrong_profiled[0, 0]
    assert correct[1, 1] > wrong_profiled[1, 1]


def test_confidence_ellipse_params_axis_aligned_case():
    # diagonal covariance -> ellipse axes line up with x/y exactly
    cov = np.diag([4.0, 1.0])  # std = 2, 1
    width, height, angle = confidence_ellipse_params(cov, n_sigma=1.0)

    assert width == pytest.approx(4.0)   # 2 * 1 * sqrt(4)
    assert height == pytest.approx(2.0)  # 2 * 1 * sqrt(1)
    assert angle % 180 == pytest.approx(0.0, abs=1e-6)


def test_confidence_ellipse_params_scales_with_n_sigma():
    cov = np.diag([4.0, 1.0])
    w1, h1, _ = confidence_ellipse_params(cov, n_sigma=1.0)
    w2, h2, _ = confidence_ellipse_params(cov, n_sigma=2.0)
    assert w2 == pytest.approx(2 * w1)
    assert h2 == pytest.approx(2 * h1)


def test_confidence_ellipse_params_rejects_wrong_shape():
    with pytest.raises(ValueError):
        confidence_ellipse_params(np.eye(3))
