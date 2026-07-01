"""
trainer.py
"""

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from training.dataset import DatasetLoader
from training.callbacks import get_callbacks

from models.cnn_bilstm_attention import build_model

from configs.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

from pathlib import Path


class Trainer:

    def __init__(self):

        loader = DatasetLoader("data/processed")

        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test
        ) = loader.split()

        # Ensure labels are integer type
        self.y_train = self.y_train.astype("int32")
        self.y_val = self.y_val.astype("int32")
        self.y_test = self.y_test.astype("int32")

        # ----------------------------
        # Debug Information
        # ----------------------------
        print("\n========== Dataset Summary ==========")

        print(f"X_train : {self.X_train.shape}")
        print(f"X_val   : {self.X_val.shape}")
        print(f"X_test  : {self.X_test.shape}")

        print(f"y_train : {self.y_train.shape}")
        print(f"y_val   : {self.y_val.shape}")
        print(f"y_test  : {self.y_test.shape}")

        print("\nUnique labels (train):", np.unique(self.y_train))
        print("Unique labels (val):", np.unique(self.y_val))
        print("Unique labels (test):", np.unique(self.y_test))

        print("Train dtype:", self.y_train.dtype)

        print("=====================================\n")

        self.model = build_model()

        print("\n========== Model Summary ==========\n")
        self.model.summary()
        print("\n===================================\n")

    def compile(self):

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )

        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\n✓ Model compiled successfully.\n")

    def train(self):

        classes = np.unique(self.y_train)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.y_train
        )

        class_weights = {
            int(classes[0]): float(weights[0]),
            int(classes[1]): float(weights[1])
        }

        print("Class Weights:", class_weights)

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(
                self.X_val,
                self.y_val
            ),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )

        Path("outputs/models").mkdir(
            parents=True,
            exist_ok=True
        )

        self.model.save(
            "outputs/models/final_model.keras"
        )

        return history