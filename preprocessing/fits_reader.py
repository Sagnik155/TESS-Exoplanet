"""
fits_reader.py

Reads TESS FITS light curves using Astropy.
"""

from pathlib import Path

from astropy.io import fits
import numpy as np


class FITSReader:

    def __init__(self, filepath):

        self.filepath = Path(filepath)

    def load(self):

        with fits.open(self.filepath) as hdul:

            data = hdul[1].data

            lightcurve = {
                "time": np.array(data["TIME"]),
                "sap_flux": np.array(data["SAP_FLUX"]),
                "pdcsap_flux": np.array(data["PDCSAP_FLUX"]),
                "flux_error": np.array(data["PDCSAP_FLUX_ERR"]),
                "quality": np.array(data["QUALITY"])
            }

        return lightcurve

    def summary(self):

        lc = self.load()

        print("=" * 60)

        print("Light Curve Summary")

        print("=" * 60)

        for key, value in lc.items():

            print(f"{key:15s} Shape: {value.shape}")

        print()

        print(f"Number of observations : {len(lc['time'])}")

        print(f"NaN values in TIME      : {np.isnan(lc['time']).sum()}")

        print(f"NaN values in FLUX      : {np.isnan(lc['pdcsap_flux']).sum()}")