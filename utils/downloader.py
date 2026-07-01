"""
downloader.py

Creates a balanced dataset and downloads one
120-second SPOC light curve for each target.

Author: Sagnik Sinha
"""

import pandas as pd

from configs.config import (
    TOI_CATALOG,
    RAW_DIR,
    POSITIVE_CLASSES,
    NEGATIVE_CLASSES,
    NUM_POSITIVE,
    NUM_NEGATIVE,
    RANDOM_STATE
)

from utils.mast_client import MASTClient


class DatasetDownloader:

    def __init__(self):

        self.catalog = pd.read_csv(
            TOI_CATALOG,
            comment="#"
        )

        self.client = MASTClient()

    def build_target_list(self):
        """
        Create a balanced list of positive
        and negative targets.
        """

        positive = self.catalog[
            self.catalog["TOI Disposition"].isin(POSITIVE_CLASSES)
        ]

        negative = self.catalog[
            self.catalog["TOI Disposition"].isin(NEGATIVE_CLASSES)
        ]

        positive = positive.sample(
            n=min(NUM_POSITIVE, len(positive)),
            random_state=RANDOM_STATE
        )

        negative = negative.sample(
            n=min(NUM_NEGATIVE, len(negative)),
            random_state=RANDOM_STATE
        )

        dataset = pd.concat(
            [positive, negative],
            ignore_index=True
        )

        dataset = dataset.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        return dataset

    def save_target_list(self, output_file):
        """
        Save selected targets.
        """

        dataset = self.build_target_list()

        dataset.to_csv(
            output_file,
            index=False
        )

        print(f"\nSaved {len(dataset)} selected targets.")

        return dataset

    def download_target(self, tic_id):
        """
        Download ONLY ONE 120-second SPOC light curve
        for this TIC.
        """

        results = self.client.search_spoc_120(tic_id)

        if len(results) == 0:

            print(f"No 120-second SPOC data found for TIC {tic_id}")

            return

        target_dir = RAW_DIR / f"TIC_{tic_id}"

        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            # Download ONLY the first available sector
            row = results[0]

            lc = self.client.download(
                row,
                target_dir
            )

            print(
                f"Downloaded Sector {lc.sector}"
            )

        except Exception as e:

            print(
                f"Download failed: {e}"
            )

    def download_dataset(self):
        """
        Download the dataset.
        """

        dataset = self.build_target_list()

        total = len(dataset)

        print("=" * 80)
        print(f"Downloading {total} targets")
        print("=" * 80)

        for index, row in dataset.iterrows():

            tic = row["TIC"]

            print(f"\n[{index+1}/{total}] TIC {tic}")

            self.download_target(tic)

        print("\nDataset download completed successfully.")