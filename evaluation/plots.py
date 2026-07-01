"""
plots.py

Creates evaluation plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_training(history):

    # Accuracy
    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Training")

    plt.plot(history.history["val_accuracy"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.savefig(FIGURE_DIR / "training_accuracy.png")

    plt.close()


    # Loss
    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Training")

    plt.plot(history.history["val_loss"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.savefig(FIGURE_DIR / "training_loss.png")

    plt.close()


def plot_confusion_matrix(cm):

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig(FIGURE_DIR / "confusion_matrix.png")

    plt.close()


def plot_pr_curve(recall, precision):

    plt.figure(figsize=(8,5))

    plt.plot(recall, precision)

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.savefig(FIGURE_DIR / "precision_recall_curve.png")

    plt.close()