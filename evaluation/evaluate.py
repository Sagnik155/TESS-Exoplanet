"""
evaluate.py

Evaluates the trained CNN-BiLSTM-Attention model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve
)

from tensorflow.keras.models import load_model

from training.dataset import DatasetLoader
from models.attention import AttentionLayer

from evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)


class Evaluator:

    def __init__(self):

        loader = DatasetLoader("data/processed")

        (
            _,
            _,
            self.X_test,
            _,
            _,
            self.y_test
        ) = loader.split()

        self.model = load_model(
            "outputs/checkpoints/best_model.keras",
            custom_objects={
                "AttentionLayer": AttentionLayer
            }
        )

    def evaluate(self):

        print("\nRunning inference...\n")

        probabilities = self.model.predict(
            self.X_test,
            verbose=0
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        accuracy = accuracy_score(
            self.y_test,
            predictions
        )

        precision = precision_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            self.y_test,
            predictions
        )

        print("=" * 40)
        print("TEST RESULTS")
        print("=" * 40)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}\n")

        print(
            classification_report(
                self.y_test,
                predictions,
                zero_division=0
            )
        )

        # -------------------------
        # Save metrics
        # -------------------------

        Path("outputs").mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            "outputs/metrics.txt",
            "w"
        ) as f:

            f.write(f"Accuracy  : {accuracy:.4f}\n")
            f.write(f"Precision : {precision:.4f}\n")
            f.write(f"Recall    : {recall:.4f}\n")
            f.write(f"F1 Score  : {f1:.4f}\n")

        # -------------------------
        # Save predictions
        # -------------------------

        results = pd.DataFrame({

            "True Label": self.y_test,

            "Predicted Label": predictions,

            "Confidence": probabilities.max(axis=1)

        })

        results.to_csv(
            "outputs/predictions.csv",
            index=False
        )

        # -------------------------
        # Plots
        # -------------------------

        plot_confusion_matrix(cm)

        precision_curve, recall_curve, _ = precision_recall_curve(
            self.y_test,
            probabilities[:, 1]
        )

        plot_pr_curve(
            recall_curve,
            precision_curve
        )

        print("\nEvaluation completed successfully.")

        print("Results saved to outputs/")