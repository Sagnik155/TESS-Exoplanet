"""
callbacks.py
"""

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

from pathlib import Path


def get_callbacks():

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        CSVLogger(
            str(log_dir / "training_log.csv")
        )

    ]

    return callbacks