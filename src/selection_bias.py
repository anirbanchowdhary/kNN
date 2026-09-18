"""
Checking whether the luminosity cut itself is entangled with a parameter.

Two distinct ways a top-fraction luminosity selection can bias a
sensitivity analysis:

1.  Whole-sim dropout: a simulation has too few luminous AGN to pass
    `MIN_AGN` and is dropped entirely. If dropout correlates with a
    parameter, the retained sample no longer spans that parameter's full
    range. `diagnose_retention_bias` checks this against the *full* LH
    parameter table (not just the retained subset -- comparing retained
    to retained tells you nothing).

2.  Within-sim confound: even when every simulation is retained, the
    *number* of AGN that survive the cut (`nbh`) can itself trend with a
    parameter (e.g. a cosmology with more structure formation produces
    more luminous AGN). `remove_abundance` already strips the kNN-CDF's
    dependence on nbh, but if nbh and a parameter are correlated, some of
    that parameter's true signal gets removed along with the abundance
    trend. `residualize_theta_on_nbh` strips the nbh-predictable part out
    of theta instead, so the quartile split downstream is not itself
    driven by nbh.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp


def diagnose_retention_bias(retained_ids, all_ids, theta_all, params=None):
    """
    Does the set of simulations that survive the luminosity cut
    (`retained_ids`, i.e. reached MIN_AGN) differ from the full LH
    parameter distribution (`all_ids`)?

    Returns a DataFrame with, per parameter: a point-biserial Spearman
    rho between retention and the parameter, and a KS test between the
    retained and full distributions. `bias_flag` is a simple, deliberately
    loose screen (not a multiple-testing-corrected significance claim).
    """
    if params is None:
        params = list(theta_all.columns)

    retained_set = set(retained_ids)
    is_retained = np.array([sid in retained_set for sid in all_ids])
    frac_retained = is_retained.mean()

    rows = []
    for p in params:
        full_vals = theta_all.loc[all_ids, p].values
        retained_vals = theta_all.loc[retained_ids, p].values

        if frac_retained > 0.999 or frac_retained < 0.001:
            rho, sp = 0.0, 1.0
        else:
            rho, sp = spearmanr(is_retained.astype(float), full_vals)

        ks_stat, ks_p = ks_2samp(retained_vals, full_vals)

        rows.append({
            "parameter": p,
            "spearman_rho": rho,
            "spearman_p": sp,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "bias_flag": abs(rho) > 0.1 or ks_p < 0.01,
        })

    return pd.DataFrame(rows)


def diagnose_nbh_confound(nbh, theta, params=None):
    """
    Among retained simulations, does the surviving AGN count `nbh`
    correlate with a parameter? (Same idea as `diagnose_retention_bias`,
    but for the within-sim effect rather than whole-sim dropout.)

    `theta` must already be aligned to `nbh` row-for-row
    (see `src.params.align_to_params`).
    """
    if params is None:
        params = list(theta.columns)

    rows = []
    for p in params:
        rho, sp = spearmanr(nbh, theta[p].values)
        rows.append({
            "parameter": p,
            "spearman_rho": rho,
            "spearman_p": sp,
            "bias_flag": sp < 0.01,
        })

    return pd.DataFrame(rows)


def residualize_theta_on_nbh(theta, nbh, params=None):
    """
    Replace each `params` column of `theta` with its residual after a
    linear fit against `nbh`, so downstream quartile splits reflect
    parameter variation that ISN'T predictable from AGN count alone.

    Use only on parameters `diagnose_nbh_confound` actually flags --
    residualizing an unconfounded parameter just adds noise for no
    benefit.
    """
    if params is None:
        params = list(theta.columns)

    nbh = np.asarray(nbh, dtype=float)
    x = np.column_stack([np.ones_like(nbh), nbh])

    theta_resid = theta.copy()
    for p in params:
        y = theta[p].values.astype(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        theta_resid[p] = y - x @ beta

    return theta_resid
