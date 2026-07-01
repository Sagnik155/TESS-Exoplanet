"""
mast_client.py

Handles communication with the MAST archive using Lightkurve.

Author: Sagnik Sinha
"""

from pathlib import Path
import lightkurve as lk


class MASTClient:
    """
    Wrapper around the MAST Lightkurve API.
    """

    def __init__(self):

        self.mission = "TESS"
        self.author = "SPOC"

    def search(self, tic_id):
        """
        Search all available SPOC light curves
        for a given TIC ID.
        """

        target = f"TIC {tic_id}"

        return lk.search_lightcurve(
            target=target,
            mission=self.mission,
            author=self.author
        )

    def search_spoc_120(self, tic_id):
        """
        Return only 120-second SPOC light curves.
        """

        result = self.search(tic_id)

        if len(result) == 0:
            return result

        mask = result.exptime.value == 120

        return result[mask]

    def inspect(self, tic_id):
        """
        Print search results.
        """

        result = self.search(tic_id)

        if len(result) == 0:
            print("No products found.")
            return

        print(result)

    def download(self, search_result, download_dir):
        """
        Download a single SearchResultRow.
        """

        lc = search_result.download(
            download_dir=str(download_dir)
        )

        return lc