"""
dataset.py

Creates TensorFlow datasets.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


class DatasetLoader:

    def __init__(self, processed_dir):

        self.processed_dir = Path(processed_dir)

    def load(self):

        X = []
        y = []

        positive = list(
            (self.processed_dir / "positive").glob("*.npy")
        )

        negative = list(
            (self.processed_dir / "negative").glob("*.npy")
        )

        for file in positive:

            X.append(np.load(file))

            y.append(1)

        for file in negative:

            X.append(np.load(file))

            y.append(0)

        X = np.array(X)

        y = np.array(y)

        X = X[..., np.newaxis]

        return X, y

    def split(self):

        X, y = self.load()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )