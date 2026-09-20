"""
Do two tracers (e.g. AGN and galaxies) carry the same clustering
information, or different information?

`sensitivity_table` answers "which parameters does each tracer respond
to" -- if both tracers are sensitive to the same parameters and nothing
else, they *could* still be redundant (both just tracing the same
large-scale structure) or genuinely complementary (each catching a
different piece of it). The direct way to tell is to look at the two
tracers' kNN-CDF fluctuations *simulation by simulation*, not just their
parameter responses: if simulation-to-simulation excursions in one
tracer's summary track excursions in the other's, the two are largely
redundant at that scale; if they don't, each is contributing information
the other lacks.
"""

import numpy as np


def bin_correlation(residuals_a, residuals_b):
    """
    Per-bin Pearson correlation between two tracers' residuals, computed
    across simulations, reshaped to (n_k, n_r) -- comparable bin-for-bin
    to `sensitivity.scale_resolved_response`.

    `residuals_a`/`residuals_b` must be the same shape (n_sims, n_k*n_r)
    and already aligned to the same simulations, row for row (e.g. both
    built from the same `sim_ids` order via `params.align_to_params`).

    A value near +-1 at some (k, r) means the two tracers move together
    there across the LH suite (redundant information); a value near 0
    means their fluctuations are independent at that scale (complementary
    information).
    """
    if residuals_a.shape != residuals_b.shape:
        raise ValueError(
            f"residuals_a and residuals_b must share a shape "
            f"(aligned to the same simulations), got "
            f"{residuals_a.shape} vs {residuals_b.shape}"
        )
    if residuals_a.shape[0] < 3:
        raise ValueError(
            f"only {residuals_a.shape[0]} paired simulations -- too few "
            f"to correlate meaningfully"
        )

    a = residuals_a - residuals_a.mean(axis=0, keepdims=True)
    b = residuals_b - residuals_b.mean(axis=0, keepdims=True)

    num = (a * b).sum(axis=0)
    den = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den

    return corr


def align_common_sims(sim_ids_a, residuals_a, sim_ids_b, residuals_b):
    """
    Restrict both residual arrays to the simulations present in *both*
    tracer runs (fixed-N selection can drop different simulations per
    tracer), in a single shared sim_id order.

    Returns (common_ids, aligned_residuals_a, aligned_residuals_b).
    """
    common = np.array(sorted(set(sim_ids_a) & set(sim_ids_b)))
    if len(common) == 0:
        raise ValueError("no simulations are common to both tracer runs")

    idx_a = {sid: i for i, sid in enumerate(sim_ids_a)}
    idx_b = {sid: i for i, sid in enumerate(sim_ids_b)}

    rows_a = [idx_a[sid] for sid in common]
    rows_b = [idx_b[sid] for sid in common]

    return common, residuals_a[rows_a], residuals_b[rows_b]
