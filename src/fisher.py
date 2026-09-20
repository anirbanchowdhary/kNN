"""
Fisher-matrix forecast: given how the kNN-CDF responds to each parameter
(the derivative) and how much it scatters at fixed parameters (the
noise), how tightly could (Omega_m, sigma_8) be constrained -- and how
does that tighten when AGN and galaxies are combined?

This is a forecast of achievable *precision*, not a predictor you feed a
single measured kNN-CDF into to get a point estimate back (that's the
regression/emulator approach in the README's roadmap). It answers "how
much information is in this statistic", not "what are Omega_m/sigma_8
for this specific simulation".

Two ingredients, both built from data already in this pipeline:

- **The response** `D[i, bin] = d(summary[bin]) / d(theta_i)`, estimated
  by ordinary least squares: for each summary bin, regress its LH
  residual against the full parameter vector (all 6 LH parameters) and
  take the coefficients. This is a genuine *partial* derivative --
  it controls for the other 5 parameters' simultaneous variation across
  the LH suite, unlike `sensitivity.scale_resolved_response`'s quartile
  contrast (which is a robust rank-based sensitivity metric, not a slope
  estimate).
- **The noise**, from a CV set's per-bin variance (`cv.per_bin_std(...)
  ** 2`) -- the pure seed-to-seed scatter at fixed parameters. Only the
  *diagonal* (per-bin variance) is used, not the full bin-to-bin
  covariance: a CV set has ~27-30 realizations, far too few to invert a
  ~150x150 empirical covariance matrix reliably. Neighbouring radial bins
  of a CDF are genuinely correlated (it's a smooth, monotone curve), and
  that correlation is usually positive, so this diagonal approximation
  likely *understates* the true uncertainty -- read every forecast here
  as optimistic, not as a rigorous bound.

Combining two tracers' Fisher matrices by simple addition (`combine_fisher`)
additionally assumes the two tracers are statistically independent given
the parameters. `complementarity.bin_correlation` measured a median
cross-tracer correlation of ~0.69 for AGN vs. galaxies on the real LH
suite -- they are *not* independent -- so a naively combined forecast
here is a best-case upper bound on how much combining helps, not the true
combined constraint. A rigorous joint forecast would need the AGN-galaxy
cross-covariance, which requires running both tracers over the *same* CV
realizations and measuring their joint scatter -- not built here.
"""

import numpy as np


def linear_response(residuals, theta, params):
    """
    Partial derivative of each summary bin w.r.t. each parameter in
    `params`, via ordinary least squares across simulations: for every
    bin, fit `residual ~ intercept + theta[params]` and keep the
    coefficient vector (all bins solved in one least-squares call).

    Parameters
    ----------
    residuals : (n_sims, n_bins) array, aligned row-for-row to `theta`
    theta     : DataFrame with at least the columns in `params`, aligned
                to `residuals` (e.g. via `params.align_to_params`)
    params    : parameter names to differentiate against, in the order
                the returned rows will follow

    Returns
    -------
    (len(params), n_bins) ndarray: D[i, j] = d(residuals[:, j]) / d(theta[params[i]])
    """
    n_sims = residuals.shape[0]
    if len(theta) != n_sims:
        raise ValueError(
            f"theta has {len(theta)} rows but residuals has {n_sims} -- not aligned"
        )

    design = np.column_stack([np.ones(n_sims)] + [theta[p].values for p in params])
    coeffs, *_ = np.linalg.lstsq(design, residuals, rcond=None)
    return coeffs[1:]  # drop the intercept row


def drop_uninformative_bins(response, bin_variance):
    """
    Drop bins where `bin_variance` is exactly zero before building a
    Fisher matrix. A CDF bin can be genuinely pinned at 0 or 1 for every
    CV realization (most common at very small/large r, where the CDF is
    saturated regardless of the seed) -- that's zero *measured* scatter
    from a finite CV sample, not infinite information, and
    `fisher_matrix` would otherwise reject the whole array over a
    handful of such bins (the same pinned-CDF effect
    `complementarity.bin_correlation` documents for its own NaN bins).

    Returns
    -------
    (response[:, mask], bin_variance[mask], n_dropped)
    """
    bin_variance = np.asarray(bin_variance)
    mask = bin_variance > 0
    return response[:, mask], bin_variance[mask], int((~mask).sum())


def fisher_matrix(response, bin_variance):
    """
    Fisher matrix from a diagonal covariance:
    `F[i, j] = sum_bin response[i, bin] * response[j, bin] / bin_variance[bin]`
    -- the diagonal-covariance special case of `F = D C^-1 D^T`.

    Parameters
    ----------
    response     : (n_params, n_bins), from `linear_response`
    bin_variance : (n_bins,), the per-bin variance of the summary vector
                   at fixed parameters (e.g. `cv.per_bin_std(...).flatten() ** 2`)

    Returns
    -------
    (n_params, n_params) ndarray
    """
    bin_variance = np.asarray(bin_variance)
    if bin_variance.shape != (response.shape[1],):
        raise ValueError(
            f"bin_variance shape {bin_variance.shape} doesn't match "
            f"response's {response.shape[1]} bins"
        )
    if np.any(bin_variance <= 0):
        raise ValueError("bin_variance must be strictly positive in every bin")

    weighted = response / bin_variance[None, :]
    return weighted @ response.T


def combine_fisher(*matrices):
    """
    Sum independent Fisher matrices -- valid only if the underlying data
    sets are statistically independent given the parameters (Fisher
    information is additive for independent data, not otherwise). See
    this module's docstring: AGN and galaxies are *not* independent in
    this project's real data, so a forecast built from this on those two
    is a best-case upper bound, not a rigorous combined constraint.
    """
    if not matrices:
        raise ValueError("combine_fisher needs at least one matrix")
    shape = matrices[0].shape
    for m in matrices:
        if m.shape != shape:
            raise ValueError(f"all Fisher matrices must share a shape, got {shape} and {m.shape}")
    return np.sum(matrices, axis=0)


def marginalized_covariance(F):
    """
    Marginalized parameter covariance = inverse Fisher matrix -- the
    correct way to get a constraint on a subset of parameters that
    properly accounts for their degeneracy with every other parameter in
    `F` (as opposed to inverting a sub-block of `F` directly, which
    implicitly assumes the other parameters are known exactly and is
    therefore too optimistic -- see `sub_covariance`).

    Raises if `F` is numerically (near-)singular rather than returning a
    meaningless inverse.
    """
    cond = np.linalg.cond(F)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(
            f"Fisher matrix is (near-)singular (condition number={cond:.2e}); "
            f"a forecast from it would not be meaningful"
        )
    return np.linalg.inv(F)


def sub_covariance(cov, params, subset):
    """
    Extract the `subset`-indexed block of a full marginalized covariance
    matrix (from `marginalized_covariance`) -- the correct way to read
    off a joint constraint on 2+ parameters out of a larger Fisher
    analysis. `params` is the full parameter list `cov`'s rows/columns
    follow; `subset` (e.g. `["Omega_m", "sigma_8"]`) selects which.

    This is deliberately not "invert `fisher_matrix`'s own subset block"
    -- that operation (profiling) gives a different, smaller-looking but
    wrong answer that ignores the subset's degeneracy with every
    parameter left out of it. Marginalize first (this function's input),
    then slice; never slice first and then invert.
    """
    idx = [params.index(p) for p in subset]
    return cov[np.ix_(idx, idx)]


def confidence_ellipse_params(cov2x2, n_sigma=1.0):
    """
    A 2x2 covariance matrix -> (width, height, angle_degrees) of its
    `n_sigma` confidence ellipse, via eigendecomposition. `width`/`height`
    are full-axis lengths (`2 * n_sigma * sqrt(eigenvalue)`); `angle` is
    the major axis's rotation from the x-axis, in degrees, suitable for
    `matplotlib.patches.Ellipse`.
    """
    cov2x2 = np.asarray(cov2x2)
    if cov2x2.shape != (2, 2):
        raise ValueError(f"expected a 2x2 covariance, got shape {cov2x2.shape}")

    vals, vecs = np.linalg.eigh(cov2x2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_sigma * np.sqrt(np.clip(vals, 0, None))
    return float(width), float(height), float(angle)
