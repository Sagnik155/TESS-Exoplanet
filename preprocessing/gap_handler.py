"""
gap_handler.py

Removes invalid observations from TESS light curves.
"""

import numpy as np


class GapHandler:

    @staticmethod
    def clean(lightcurve):

        time = lightcurve["time"]
        flux = lightcurve["pdcsap_flux"]
        error = lightcurve["flux_error"]
        quality = lightcurve["quality"]

        valid = (
            np.isfinite(time)
            & np.isfinite(flux)
            & np.isfinite(error)
            & (quality == 0)
        )

        cleaned = {
            "time": time[valid],
            "flux": flux[valid],
            "error": error[valid]
        }

        return cleaned