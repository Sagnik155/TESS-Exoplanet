from pathlib import Path

from utils.downloader import DatasetDownloader

from configs.config import LABELS_DIR


def main():

    downloader = DatasetDownloader()

    selected_file = LABELS_DIR / "selected_targets.csv"

    downloader.save_target_list(
        selected_file
    )

    downloader.download_dataset()


if __name__ == "__main__":

    main()
    