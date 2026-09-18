"""
Removing the trivial "more AGN -> different CDF" trend.

The number of AGN surviving the luminosity cut (`nbh`) varies from
simulation to simulation and, on its own, shifts every kNN-CDF bin --
more tracers means closer typical nearest neighbors, independent of any
interesting clustering physics. Regressing each bin on log10(nbh) and
keeping the residual isolates the part of the signal that isn't just
"how many AGN did this simulation happen to have".
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def remove_abundance(summaries, nbh):
    """
    Parameters
    ----------
    summaries : (n_sims, n_bins) kNN-CDF summaries
    nbh       : (n_sims,) AGN counts, same row order as summaries

    Returns
    -------
    residuals : (n_sims, n_bins), each column's linear trend against
                log10(nbh) removed
    """
    x = np.log10(nbh).reshape(-1, 1)
    residuals = np.zeros_like(summaries, dtype=np.float64)

    for j in range(summaries.shape[1]):
        y = summaries[:, j]
        model = LinearRegression().fit(x, y)
        residuals[:, j] = y - model.predict(x)

    return residuals
