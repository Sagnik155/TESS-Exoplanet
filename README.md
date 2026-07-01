# Exoplanet AI
### AI-Powered Exoplanet Detection Using TESS Light Curves

An end-to-end deep learning pipeline for automated exoplanet detection using NASA TESS observations. This project retrieves astronomical light curves from the MAST archive, preprocesses the data, and classifies potential exoplanet transit signals using a hybrid CNN–BiLSTM–Attention neural network.

Developed for the **ISRO Bharatiya Antariksh Hackathon**.

---

# Overview

Modern astronomical missions such as NASA's Transiting Exoplanet Survey Satellite (TESS) generate enormous volumes of stellar observations every day. Identifying planetary transits from these observations manually is both time-consuming and computationally expensive.

This project automates the complete workflow:

- Retrieve observations from the NASA MAST Archive
- Download TESS SPOC light curves
- Preprocess astronomical time-series data
- Train a deep learning model
- Detect potential exoplanets
- Evaluate model performance

---

# Features

- Automatic retrieval of TESS observations
- Direct integration with NASA MAST Archive
- FITS file processing
- Light curve preprocessing pipeline
- CNN + BiLSTM + Attention architecture
- Automated exoplanet classification
- Performance evaluation and visualization
- Research-ready dataset generation

---

# Project Architecture

```
NASA MAST Archive
        │
        ▼
 Download TESS Data
        │
        ▼
   FITS Processing
        │
        ▼
 Data Cleaning
        │
        ▼
Normalization
        │
        ▼
Sequence Preparation
        │
        ▼
 CNN Feature Extraction
        │
        ▼
 BiLSTM Temporal Learning
        │
        ▼
 Attention Mechanism
        │
        ▼
 Softmax Classifier
        │
        ▼
Planet Candidate / False Positive
```

---

# Repository Structure

```
Exoplanet/

│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
│
├── evaluation/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── preprocessing/
│
├── scripts/
│
├── training/
│
├── utils/
│
├── main.py
│
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Sagnik155/TESS-Exoplanet.git

cd TESS-Exoplanet
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Download Dataset

```bash
python -m scripts.download_dataset
```

---

# Preprocess Dataset

```bash
python -m scripts.preprocess_dataset
```

---

# Train Model

```bash
python -m scripts.train_model
```

---

# Evaluate Model

```bash
python -m scripts.evaluate_model
```

---

# Model Architecture

The proposed model combines spatial and temporal feature extraction:

- 1D Convolutional Neural Network (CNN)
- Bidirectional LSTM
- Attention Layer
- Fully Connected Layers
- Softmax Output

This hybrid architecture enables robust extraction of planetary transit features from stellar light curves.

---

# Data Source

This project uses publicly available observations from:

- NASA TESS Mission
- MAST Archive
- SPOC Light Curves

No proprietary datasets are required.

---

# Tech Stack

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras

### Scientific Libraries

- NumPy
- Pandas
- Astropy
- Lightkurve
- Scikit-learn

### Visualization

- Matplotlib

---

# Workflow

```
Retrieve TESS Data
        │
        ▼
Download FITS Files
        │
        ▼
Preprocess Light Curves
        │
        ▼
Create Dataset
        │
        ▼
Train CNN-BiLSTM-Attention
        │
        ▼
Evaluate Model
        │
        ▼
Predict Exoplanet Candidate
```

---

# Future Improvements

- Web-based prediction dashboard
- Multi-class exoplanet classification
- Explainable AI (Grad-CAM / SHAP)
- Real-time MAST querying
- Cloud deployment
- Automated report generation
- Support for additional astronomical missions

---

# Authors

**Sagnik Sinha**

Developed as part of the **ISRO Bharatiya Antariksh Hackathon**.

---

# License

This project is intended for educational and research purposes.

---

# Acknowledgements

- NASA
- TESS Mission
- MAST Archive
- Lightkurve Team
- Astropy Collaboration
- TensorFlow Team
