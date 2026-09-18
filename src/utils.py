import numpy as np
from scipy.interpolate import interp1d


def cdf_on_grid(distances, rgrid):

    rs = np.sort(distances)

    Fs = np.arange(
        1,
        len(rs)+1
    )/len(rs)

    return interp1d(
        rs,
        Fs,
        bounds_error=False,
        fill_value=(0,1)
    )(rgrid)