from utils.downloader import DatasetDownloader

downloader = DatasetDownloader()

targets = downloader.build_target_list()

print(targets[["TIC", "TOI Disposition"]].head())

print()

print(targets["TOI Disposition"].value_counts())

downloader.save_target_list(
    "data/labels/selected_targets.csv"
)