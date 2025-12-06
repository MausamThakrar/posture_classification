# Posture Classification Using Machine Learning and Deep Learning Techniques  
### Master’s Thesis Project – Mausam Thakrar  

---

## Overview

This repository contains the full implementation of my **Master’s thesis project**,  
which focuses on developing an intelligent system to classify human sitting posture  
using sensor-based data from a smart chair.

The project implements a complete **end-to-end machine learning pipeline**:

- Data preprocessing (noise reduction, DBSCAN outlier removal)
- Feature scaling and label encoding
- Classical ML models (Logistic Regression, Random Forest, SVM)
- Deep learning models (CNN, RNN)
- Evaluation (accuracy, classification reports, confusion matrices)
- A small **Streamlit dashboard** for interactive posture prediction
- Jupyter notebooks for EDA, feature engineering and analysis

---

## Repository Structure

```text
posture-classification-thesis/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── sample_posture_data.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_results_visualization.ipynb
│   └── 05_noise_outlier_analysis.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── noise_reduction.py
│   │   ├── dbscan_outlier_removal.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── svm.py
│   │   ├── rf.py
│   │   ├── mnr.py
│   │   ├── cnn.py
│   │   ├── rnn.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── accuracy_plot.py
│   │   ├── precision_recall_plot.py
│   │   └── __init__.py
│   │
│   └── train_pipeline.py
│
└── app/
    └── streamlit_app.py
