"""
preprocessing_pipeline.py

Complete preprocessing pipeline for TESS light curves.
"""

from preprocessing.fits_reader import FITSReader
from preprocessing.gap_handler import GapHandler
from preprocessing.detrending import Detrender
from preprocessing.normalization import Normalizer


class PreprocessingPipeline:

    def __init__(self, filepath):

        self.filepath = filepath

    def run(self):

        # Step 1
        reader = FITSReader(self.filepath)

        lightcurve = reader.load()

        # Step 2
        lightcurve = GapHandler.clean(lightcurve)

        # Step 3
        lightcurve = Detrender.detrend(lightcurve)

        # Step 4
        lightcurve = Normalizer.normalize(lightcurve)

        return lightcurve