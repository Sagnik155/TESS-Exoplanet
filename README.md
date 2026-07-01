# ExoplanetDetection

Project scaffold for exoplanet detection using astronomical time-series data.

## Structure

- `configs/` - configuration settings
- `data/` - raw, processed, and split datasets
- `preprocessing/` - data loading and preprocessing utilities
- `models/` - model definitions and loss functions
- `training/` - training loop and callbacks
- `evaluation/` - evaluation metrics and plots
- `utils/` - helper utilities
- `outputs/` - generated logs, saved models, and figures
- `notebooks/` - exploratory notebooks

## Getting Started

1. Create or place your FITS data in `data/raw/fits/positive` and `data/raw/fits/negative`
2. Add your TOI catalog to `data/labels/toi_catalog.csv`
3. Run preprocessing, training, and evaluation scripts as needed
