"""
dataset_builder.py

Creates model-ready numpy arrays.
"""

from pathlib import Path
import numpy as np

from configs.config import (
    PROCESSED_DIR,
    POSITIVE_CLASSES,
    SEQUENCE_LENGTH
)


class DatasetBuilder:

    @staticmethod
    def resize(signal):

        signal = np.asarray(signal, dtype=np.float32)

        n = len(signal)

        if n > SEQUENCE_LENGTH:

            idx = np.linspace(
                0,
                n - 1,
                SEQUENCE_LENGTH
            ).astype(int)

            signal = signal[idx]

        elif n < SEQUENCE_LENGTH:

            padded = np.zeros(
                SEQUENCE_LENGTH,
                dtype=np.float32
            )

            padded[:n] = signal

            signal = padded

        return signal

    @staticmethod
    def save(signal, tic, sector, disposition):

        folder = (
            PROCESSED_DIR / "positive"
            if disposition in POSITIVE_CLASSES
            else PROCESSED_DIR / "negative"
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = folder / f"TIC_{tic}_S{sector}.npy"

        np.save(filename, signal)

        return filename