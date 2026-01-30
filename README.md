# Two-Tier Lightweight IoT Intrusion Detection System (IDS)

This repository contains the implementation of a **two-tier lightweight IDS for IoT/IIoT networks**, developed as part of PhD research.

**Architecture (core idea):**
- **Edge tier (lightweight decision):** BBFS *(wrapper)* feature selection + **Logistic Regression** (CPU-only, scikit-learn)
- **Cloud tier (high-capacity refinement):** BBFS *(filter)* feature selection + **HGS-Optimized 1D-CNN** (TensorFlow CPU)

The design targets **resource-constrained edge environments** while enabling **stronger cloud-level detection** using optimized deep learning.

---

## Pipeline Overview

### 1) Data Preparation (Interactive)
- Dataset selection from `data/raw` (or browse path)
- Optional **stratified fraction loading** (demo mode)
- Column inspection + additional drop columns
- Train/Test split (stratified)
- Normalization (fit on train, transform test)
- Optional balancing on **train only**:
  - SMOTE
  - SMOTE + ENN
  - SMOTE + ENN + LOF

### 2) Edge Tier: BBFS-Wrapper + Logistic Regression
- Wrapper-based BBFS searches feature subsets using a Logistic Regression evaluator
- Prints **iteration-wise and agent-wise** feature subsets and fitness
- Produces a selected feature set for lightweight edge inference

### 3) Cloud Tier: BBFS-Filter + HGS-Optimized 1D-CNN
- Filter-based BBFS using:
  - Conditional Mutual Information (CMI)
  - Fisher score
  - Correlation penalty
- Selected features are passed to an **HGS-optimized 1D-CNN** training routine
- Designed for stronger detection capacity at the cloud tier

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

### Edge Tier (BBFS-Wrapper + Logistic Regression)
```bash
python scripts/edge/edge_bbfs_lr.py
```

### Cloud Tier (BBFS-Filter + HGS-Optimized 1D-CNN)
```bash
python scripts/cloud/cloud_hgs_cnn_train.py
```

Each script is **interactive** and guides you through:
- Loading NPZ **or** running raw-data preprocessing
- Feature selection (BBFS wrapper/filter)
- Training + evaluation

---

## Notes (Important)
- **Datasets are not included** due to size/licensing constraints.
- Prepared arrays (`.npz`), trained models, and large artifacts should remain excluded via `.gitignore`.
- The implementation is **CPU-compatible** (no CUDA required).

---

## Reproducibility
- Random seeds are configurable in interactive prompts
- Intermediate outputs can be saved as `.npz`
- The two tiers are modular and can be executed independently

---

## Contact
For academic collaboration/questions: **satya.cnis@gmail.com**

---

## License
MIT (see `LICENSE`).

---

## Citation
If you use this repository in academic work, please cite it using the metadata in `CITATION.cff`.