# Lightweight IoT Intrusion Detection System

This repository contains the implementation of a **multi-level lightweight Intrusion Detection System (IDS)** for IoT/IIoT networks, developed as part of PhD research.

The project emphasizes **feature efficiency, modular design, and reproducibility**, targeting both **edge** and **cloud** deployment scenarios.

---

## Pipeline Overview

### 1. Data Preparation
- Interactive preprocessing
- Dataset selection from `data/raw`
- Stratified sampling (full or fractional)
- Feature normalization (train-only fit)
- Class imbalance handling:
  - SMOTE
  - SMOTE + ENN
  - SMOTE + ENN + LOF

### 2. Edge Layer (Lightweight IDS)
- Bowerbird Courtship-Inspired Feature Selection (BBFS – Wrapper)
- Logistic Regression (CPU-only, scikit-learn)
- Designed for low-resource IoT edge devices

### 3. Cloud Layer (Advanced IDS)
- BBFS Filter using:
  - Conditional Mutual Information (CMI)
  - Fisher Score
  - Correlation Penalty
- Base 1D-CNN for traffic classification
- HGS-Optimized 1D-CNN (Hunger Games Search for hyperparameter tuning)

---

## Project Structure

```
lightweight-iot-ids/
│
├── scripts/
│   ├── data_prep_interactive.py
│   ├── edge/
│   │   └── edge_bbfs_lr.py
│   └── cloud/
│       ├── cloud_bbfs_filter.py
│       ├── cloud_cnn_train.py
│       └── cloud_hgs_cnn_train.py
│
├── data/
│   └── raw/              # Datasets (not tracked in Git)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Edge-Level IDS
```bash
python scripts/edge/edge_bbfs_lr.py
```

### Cloud-Level CNN
```bash
python scripts/cloud/cloud_cnn_train.py
```

### Cloud-Level HGS-Optimized CNN
```bash
python scripts/cloud/cloud_hgs_cnn_train.py
```

Each script is **interactive** and guides the user through:
- Dataset loading
- Preprocessing
- Feature selection
- Model training and evaluation

---

## Notes

- Datasets are **not included** due to size and licensing constraints.
- Prepared data (`.npz`), trained models, and plots are excluded via `.gitignore`.
- All experiments are CPU-compatible (no CUDA required).

---

## Reproducibility

- Random seeds are configurable
- Intermediate outputs can be saved as `.npz`
- Modular scripts allow independent execution of each pipeline stage

---

## License

This project is intended for academic and research use.  
Add a license file (e.g., MIT, Apache 2.0) if redistribution is planned.

---

## Citation

If you use this code in your research, please cite the corresponding journal article or thesis (to be updated).