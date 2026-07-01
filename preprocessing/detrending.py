"""
detrending.py

Custom detrending for TESS light curves.
"""

import numpy as np
from scipy.signal import savgol_filter


class Detrender:

    @staticmethod
    def detrend(lightcurve,
                window_length=301,
                polyorder=2):

        flux = lightcurve["flux"]

        # Window length must be odd
        if window_length % 2 == 0:
            window_length += 1

        # Window cannot exceed signal length
        if window_length >= len(flux):
            window_length = len(flux) - 1

            if window_length % 2 == 0:
                window_length -= 1

        trend = savgol_filter(
            flux,
            window_length=window_length,
            polyorder=polyorder
        )

        detrended_flux = flux - trend

        lightcurve["flux"] = detrended_flux

        return lightcurve