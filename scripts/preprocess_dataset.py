"""
Processes every downloaded FITS file.
"""

from pathlib import Path
import re

import pandas as pd

from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.dataset_builder import DatasetBuilder

from configs.config import TOI_CATALOG


catalog = pd.read_csv(
    TOI_CATALOG,
    comment="#"
)

label_map = dict(
    zip(
        catalog["TIC"],
        catalog["TOI Disposition"]
    )
)

fits_files = list(
    Path("data/raw").rglob("*.fits")
)

print(f"\nFound {len(fits_files)} FITS files.\n")

for fits_file in fits_files:

    try:

        pipeline = PreprocessingPipeline(fits_file)

        lightcurve = pipeline.run()

        signal = DatasetBuilder.resize(
            lightcurve["flux"]
        )

        tic_match = re.search(
            r"TIC_(\d+)",
            str(fits_file)
        )

        sector_match = re.search(
            r"s(\d{4})",
            fits_file.name
        )

        if tic_match is None:

            continue

        tic = int(
            tic_match.group(1)
        )

        sector = (
            sector_match.group(1)
            if sector_match
            else "0000"
        )

        disposition = label_map.get(
            tic,
            "FP"
        )

        output = DatasetBuilder.save(
            signal,
            tic,
            sector,
            disposition
        )

        print(output)

    except Exception as e:

        print(f"\nSkipped {fits_file}")

        print(e)

print("\nFinished preprocessing.")