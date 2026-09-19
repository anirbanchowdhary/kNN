"""
Project-wide constants for the AGN kNN-CDF sensitivity pipeline.

Everything here mirrors the CAMELS-IllustrisTNG L25n256 Latin-Hypercube
(LH) suite: a 25 Mpc/h periodic box, 1000 simulations each drawn from a
different (Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2) combination.
"""

import numpy as np

# ----------------------------------------------------------------------
# Simulation / box
# ----------------------------------------------------------------------

BOXSIZE = 25.0          # Mpc/h, periodic

SNAP = 50                # snapshot number analyzed (single-snapshot v1)

# ----------------------------------------------------------------------
# AGN (luminosity-selected BH) selection
# ----------------------------------------------------------------------

MASS_CUT = 1e6            # Msun, base BH-mass floor before luminosity cut
TOP_FRACTION = 0.10       # keep top 10% by bolometric luminosity
MIN_AGN = 5                # minimum retained AGN for a sim to be usable

EPS_RAD = 0.1               # radiative efficiency
MSUN = 1.98847e33            # g
YEAR = 3.15576e7               # s
C_LIGHT = 2.99792458e10          # cm/s
MDOT_UNIT_TO_MSUN_YR = 10.22       # BH_Mdot (code units) -> Msun/yr

# ----------------------------------------------------------------------
# kNN-CDF
# ----------------------------------------------------------------------

KVALS = (1, 2, 4)

NRAND = 100_000            # random query points for the kNN-CDF

RGRID = np.logspace(-1.5, 1.2, 50)   # Mpc/h, radial grid the CDFs are read on

RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

COSMO_PARAMS = ["Omega_m", "sigma_8"]
ASTRO_PARAMS = ["A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
ALL_PARAMS = COSMO_PARAMS + ASTRO_PARAMS

# ----------------------------------------------------------------------
# Paths (override these to point at your local data)
# ----------------------------------------------------------------------
#
# CAMELS data on disk is organized Type / Suite / Generation / Set / Realization
# (https://camels.readthedocs.io/en/latest/organization.html):
#   Sims/IllustrisTNG/L25n256/LH/LH_<0..999>/snapshot_<###>.hdf5
#   Parameters/IllustrisTNG/L25n256/LH/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
# DATA_ROOT should point at the directory containing the "Sims" and
# "Parameters" type folders (e.g. a Globus/URL download root, or the
# Rusty Cluster PUBLIC_RELEASE mount).

SUITE = "IllustrisTNG"
GENERATION = "L25n256"   # 25 Mpc/h box, 256^3 particles -- matches BOXSIZE above
SET_NAME = "LH"            # Latin Hypercube: LH_0 .. LH_999

DATA_ROOT = "../../Data"

SIM_PATH = f"{DATA_ROOT}/Sims/{SUITE}/{GENERATION}/{SET_NAME}"
PARAMS_FILE = (
    f"{DATA_ROOT}/Parameters/{SUITE}/{GENERATION}/{SET_NAME}/"
    f"CosmoAstroSeed_{SUITE}_{GENERATION}_{SET_NAME}.txt"
)
OUTPUT_DIR = "../outputs"
