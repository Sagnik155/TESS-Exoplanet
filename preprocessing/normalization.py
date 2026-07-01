"""
normalization.py

Normalizes detrended flux values.
"""

import numpy as np


class Normalizer:

    @staticmethod
    def normalize(lightcurve):

        flux = lightcurve["flux"]

        median = np.median(flux)
        std = np.std(flux)

        if std == 0:
            std = 1.0

        flux = (flux - median) / std

        lightcurve["flux"] = flux

        return lightcurve