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

SIM_PATH = "../../Data/Sims/IllustrisTNG/LH"
PARAMS_FILE = (
    "../CAMELS-master/docs/params/IllustrisTNG/"
    "CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
)
OUTPUT_DIR = "../outputs"
