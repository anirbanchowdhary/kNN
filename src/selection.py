"""
Selecting luminosity-selected AGN from a black-hole catalog.

Bolometric luminosity is derived from the accretion rate (BH_Mdot) using
the standard efficiency relation L_bol = eps_rad * Mdot*c^2, and compared
to the Eddington luminosity only to report fedd for diagnostics; the
selection itself is a straight top-fraction cut on L_bol among BHs above
a base mass floor.
"""

import numpy as np

from .config import EPS_RAD, MSUN, YEAR, C_LIGHT, MDOT_UNIT_TO_MSUN_YR


def bolometric_luminosity(bh_mdot):
    """BH_Mdot (code units) -> L_bol in erg/s."""
    mdot_msun_yr = bh_mdot * MDOT_UNIT_TO_MSUN_YR
    mdot_cgs = mdot_msun_yr * MSUN / YEAR
    return EPS_RAD * mdot_cgs * C_LIGHT**2


def eddington_ratio(bh_mass, lbol):
    """L_bol / L_edd, with L_edd = 1.26e38 erg/s per solar mass."""
    ledd = 1.26e38 * bh_mass
    return lbol / ledd


def mass_eligible_mask(bh_mass, mass_cut):
    """BHs above the mass floor -- no luminosity requirement."""
    return bh_mass > mass_cut


def select_most_massive_n(bh_mass, mass_cut, n_target):
    """
    Boolean mask selecting exactly the `n_target` most massive BHs above
    `mass_cut`.

    The mass-selected counterpart to `select_brightest_n`: same fixed-N
    contract (all-False when fewer than `n_target` are eligible), so it can
    be swapped in for a controlled comparison at identical tracer density --
    "does luminosity selection add clustering information beyond host-halo
    mass, which Omega_m sets and feedback mostly doesn't?".
    """
    eligible = mass_eligible_mask(bh_mass, mass_cut)

    if eligible.sum() < n_target:
        return np.zeros(len(bh_mass), dtype=bool)

    idx = np.flatnonzero(eligible)
    most_massive = idx[np.argsort(bh_mass[idx])[::-1][:n_target]]

    mask = np.zeros(len(bh_mass), dtype=bool)
    mask[most_massive] = True
    return mask


def eligible_mask(bh_mass, bh_mdot, mass_cut):
    """BHs above the mass floor with a finite, positive luminosity."""
    lbol = bolometric_luminosity(bh_mdot)
    return (bh_mass > mass_cut) & np.isfinite(lbol) & (lbol > 0)


def count_eligible(bh_mass, bh_mdot, mass_cut):
    """How many BHs could be selected from, before any luminosity ranking."""
    return int(eligible_mask(bh_mass, bh_mdot, mass_cut).sum())


def select_brightest_n(bh_mass, bh_mdot, mass_cut, n_target):
    """
    Boolean mask selecting exactly the `n_target` most luminous eligible BHs.

    This is the fixed-number-density selection: every simulation contributes
    the same number of tracers, so the kNN-CDF's (strong, nonlinear)
    dependence on tracer number density is held fixed by construction rather
    than regressed out afterwards. Any remaining parameter response is then
    clustering, not abundance.

    Returns
    -------
    mask : bool ndarray. All-False when fewer than `n_target` BHs are
           eligible -- such a simulation cannot contribute at this density
           and must be dropped (which makes whole-sim retention bias the
           thing to watch; see selection_bias.diagnose_retention_bias).
    """
    eligible = eligible_mask(bh_mass, bh_mdot, mass_cut)

    if eligible.sum() < n_target:
        return np.zeros(len(bh_mass), dtype=bool)

    lbol = bolometric_luminosity(bh_mdot)
    idx = np.flatnonzero(eligible)
    brightest = idx[np.argsort(lbol[idx])[::-1][:n_target]]

    mask = np.zeros(len(bh_mass), dtype=bool)
    mask[brightest] = True
    return mask


def select_luminous_agn(bh_mass, bh_mdot, mass_cut, top_fraction):
    """
    Boolean mask selecting the top `top_fraction` of BHs by bolometric
    luminosity, restricted to BHs above `mass_cut`.

    Returns
    -------
    mask : bool ndarray, same length as bh_mass/bh_mdot
           All-False if fewer than 20 BHs pass the base mass cut, or
           fewer than 20 have a finite, positive luminosity (too small a
           sample to define a percentile cut meaningfully).
    """
    n = len(bh_mass)
    base = bh_mass > mass_cut

    if base.sum() < 20:
        return np.zeros(n, dtype=bool)

    lbol = bolometric_luminosity(bh_mdot)
    good = base & np.isfinite(lbol) & (lbol > 0)

    if good.sum() < 20:
        return np.zeros(n, dtype=bool)

    lcut = np.percentile(lbol[good], 100 * (1 - top_fraction))
    return good & (lbol >= lcut)
