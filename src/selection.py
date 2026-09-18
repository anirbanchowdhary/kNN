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
