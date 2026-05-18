# 🛡️ Detecting Fraudulent Insurance Claims Using Data Mining Techniques

A hybrid machine learning framework for detecting fraudulent insurance claims, combining Random Forest classification with anomaly detection and SMOTE-based class balancing.

> **Author:** Laiba Imran  
> **Institution:** FAST School of Management Sciences, FAST National University of Computing and Emerging Sciences, Islamabad, Pakistan

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Compared](#models-compared)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

Insurance fraud causes billions of dollars in annual losses globally, driving up premiums for legitimate policyholders. Traditional rule-based and manual detection systems struggle to handle large-scale, complex claim data effectively.

This project proposes a **hybrid fraud detection framework** that integrates:
- Data preprocessing and feature engineering
- SMOTE-based class imbalance handling
- Random Forest ensemble classification
- Isolation Forest anomaly detection

The framework achieves **87% accuracy**, **88% precision**, and **87% recall**, significantly outperforming baseline models.

---

## Key Features

- **Hybrid detection approach** — combines supervised learning (Random Forest) with unsupervised anomaly detection (Isolation Forest)
- **Class imbalance handling** — applies SMOTE to generate synthetic minority class samples
- **Comprehensive preprocessing pipeline** — handles missing values, encoding, normalization, and duplicate removal
- **Comparative evaluation** — benchmarks against Logistic Regression and Isolation Forest baselines
- **Low false positive rate** — reduces unnecessary investigation costs for insurers

---

## Dataset

- **Source:** Publicly available insurance fraud dataset on [Kaggle](https://www.kaggle.com/)
- **Contents:** Structured insurance claim records including:
  - Policyholder demographics
  - Vehicle information
  - Incident descriptions
  - Claim-related attributes
- **Target variable:** `fraud_reported` (binary: `1` = Fraudulent, `0` = Legitimate)
- **Challenge:** Significant class imbalance — fraudulent claims are a small minority

---

## Methodology

```
Data Collection → Preprocessing → Feature Engineering → SMOTE → Model Training → Evaluation
```

1. **Data Preprocessing** — Handle missing values, remove duplicates, encode categorical features, normalize numerical features
2. **Feature Engineering** — Remove irrelevant attributes, derive new features to capture hidden relationships
3. **Class Imbalance Handling** — Apply SMOTE via `imbalanced-learn` to balance the training set
4. **Model Training** — Train Random Forest (100 trees) on the balanced dataset; 80/20 train-test split
5. **Anomaly Detection** — Isolation Forest (contamination rate = 0.1) for detecting unusual claim patterns
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve

---

## Models Compared

| Model | Role |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Proposed model (primary classifier) |
| Isolation Forest | Anomaly detection component |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.55 | 0.58 | 0.57 | 0.57 |
| **Random Forest** | **0.87** | **0.88** | **0.87** | **0.87** |
| Isolation Forest | 0.41 | 0.12 | 0.02 | 0.03 |

- **ROC-AUC:** 0.94
- Random Forest outperforms all baselines across every metric
- SMOTE significantly improved recall on the minority (fraud) class

---

## Installation

**Prerequisites:** Python 3.7+

```bash
# Clone the repository
git clone https://github.com/your-username/insurance-fraud-detection.git
cd insurance-fraud-detection

# Install required libraries
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

---

## Usage

The project is designed to run in **Google Colab** or any standard Python environment.

```python
# 1. Load and preprocess the dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 2. Apply SMOTE for class balancing
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# 3. Train the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# 4. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, rf_model.predict(X_test)))
```

---

## Project Structure

```
insurance-fraud-detection/
│
├── data/
│   └── insurance_claims.csv       # Kaggle dataset (download separately)
│
├── notebooks/
│   └── fraud_detection.ipynb      # Main Google Colab notebook
│
├── src/
│   ├── preprocessing.py           # Data cleaning and encoding
│   ├── feature_engineering.py     # Feature selection and transformation
│   ├── model_training.py          # Model training and SMOTE
│   └── evaluation.py              # Metrics, confusion matrix, ROC curve
│
├── results/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   ├── precision_recall.png
│   ├── f1_score_comparison.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

## Limitations

- Trained on a single publicly available dataset — generalizability to other domains may be limited
- SMOTE generates synthetic samples that may not always reflect real fraud patterns
- Random Forest lacks interpretability compared to simpler models
- No real-time fraud detection capability in the current implementation
- Isolation Forest performed poorly as a standalone classifier for this structured dataset

---

## Future Work

- Evaluate on larger, more diverse datasets to improve generalizability
- Integrate **deep learning** and **graph-based** approaches (e.g., GNN) for improved pattern detection
- Incorporate **Explainable AI (XAI)** techniques (e.g., SHAP) to improve model interpretability
- Develop a **real-time fraud detection** pipeline for immediate flagging of suspicious claims
- Explore better integration strategies for combining supervised and unsupervised models

---

## 📄 Citation

If you use this work, please cite:

```
Laiba Imran, "Detecting Fraudulent Insurance Claims Using Data Mining Techniques,"
FAST School of Management Sciences, FAST-NUCES, Islamabad, Pakistan.
```

---

## 📬 Contact

**Laiba Imran**  
📧 laibaimran484@gmail.com  
🏛️ FAST National University of Computing and Emerging Sciences, Islamabad
