"""
config.py

Central configuration file for the Exoplanet Detection project.
"""

from pathlib import Path

# -------------------------------------------------
# Project Directories
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

LABELS_DIR = DATA_DIR / "labels"

TRAIN_DIR = DATA_DIR / "train"

VALIDATION_DIR = DATA_DIR / "validation"

TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MODEL_DIR = OUTPUT_DIR / "models"

FIGURE_DIR = OUTPUT_DIR / "figures"

LOG_DIR = OUTPUT_DIR / "logs"

# -------------------------------------------------
# Dataset Configuration
# -------------------------------------------------

TOI_CATALOG = LABELS_DIR / "toi_catalog.csv"

POSITIVE_CLASSES = ["CP", "KP", "PC"]

NEGATIVE_CLASSES = ["FP", "EB"]

NUM_POSITIVE = 100

NUM_NEGATIVE = 100

SEQUENCE_LENGTH = 2048

RANDOM_STATE = 42

# -------------------------------------------------
# Training Configuration
# -------------------------------------------------

TRAIN_SPLIT = 0.80

VALIDATION_SPLIT = 0.10

TEST_SPLIT = 0.10

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-3


# =====================================
# Model Configuration
# =====================================

SEQUENCE_LENGTH = 2048

NUM_CLASSES = 2

LEARNING_RATE = 1e-3

BATCH_SIZE = 16

EPOCHS = 30

DROPOUT = 0.3