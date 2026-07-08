# Credit Card Fraud Investigation System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## Overview

This project builds a complete fraud-detection pipeline on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — from raw transaction data to a deployable scoring tool. Fraud detection is fundamentally a rare-event problem: only 492 of the 284,807 transactions in this dataset are fraudulent (0.17%), which makes accuracy meaningless as a metric — a model that never flags fraud would still score 99.8%. The pipeline trains five classifiers with class-weighted balancing and selects the best one using PR-AUC, a metric far more sensitive to minority-class performance than ROC-AUC or raw accuracy. The winning model is then served through a Streamlit app that scores individual transactions and returns a risk rating.

## Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)
- 284,807 transactions made by European cardholders over two days in September 2013
- 492 labeled as fraud (0.172% of the data)
- `V1`–`V28` are PCA-transformed components (anonymized for confidentiality); `Time` and `Amount` are the only raw features
- The full `creditcard.csv` isn't committed to this repo due to size — see [Retrain from scratch](#retrain-from-scratch) below
- `data/demo_transactions.csv` is a 40,000-row sample (104 fraud cases) bundled only for the Streamlit demo, not used in training

## Pipeline

`train.py` runs the full training workflow:

- Loads the raw transaction data and performs a stratified 80/20 train-test split, preserving the fraud ratio in both sets
- Scales only `Time` and `Amount` with `StandardScaler` — the remaining 28 features are already PCA components and don't need it
- Trains five classifiers, each with `class_weight="balanced"` to offset the imbalance directly rather than resampling the data: Logistic Regression, Decision Tree, Random Forest, Extra Trees, and Hist Gradient Boosting
- Scores every model on precision, recall, F1, PR-AUC, and ROC-AUC, and picks a winner ranked by `(PR-AUC, recall, F1)`
- Serializes the winning model, the fitted scaler, and the feature column order to `models/`, and writes the comparison table to `RESULTS.md`

## Results

The table below is what `train.py` writes to `RESULTS.md` after each run:

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.7159 | 0.9722 |
| Decision Tree | 0.6762 | 0.7245 | 0.6995 | 0.4904 | 0.8619 |
| Random Forest | 0.8953 | 0.7857 | 0.8370 | 0.8655 | 0.9580 |
| **Extra Trees** | **0.9512** | 0.7959 | **0.8667** | **0.8763** | 0.9533 |
| Hist Gradient Boosting | 0.3398 | 0.8980 | 0.4930 | 0.7449 | 0.9579 |

**Extra Trees** was selected — best PR-AUC and precision, with recall competitive against the other ensembles. A couple of things stand out in the table itself:

- Logistic Regression posts the highest ROC-AUC of the five (0.9722), but a precision of 0.061 makes it unusable in practice — at the default threshold it flags far too many genuine transactions as fraud. It's a clean illustration of why ROC-AUC alone is misleading on heavily imbalanced data.
- Hist Gradient Boosting shows the same pattern on a smaller scale — a respectable PR-AUC undercut by weak precision — which suggests its default 0.5 decision threshold may be poorly calibrated for this class distribution, rather than a fundamental limitation of the model.

## Feature Importance

`V1`–`V28` can't be mapped back to real transaction attributes, but their relative ranking still shows what the model leans on most:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | V14 | 0.1350 |
| 2 | V4 | 0.1215 |
| 3 | V17 | 0.1109 |
| 4 | V12 | 0.1014 |
| 5 | V11 | 0.0847 |

`V14`, `V4`, and `V17` alone account for over a third of the model's total feature importance.

## Demo App

`app.py` is a Streamlit interface for scoring individual transactions:

- The public dataset has no real transaction IDs, so the app uses row index as a stand-in, and lists sample fraud row indexes to try
- Selecting a row and clicking **Analyze Transaction** runs it through the saved model and scaler, returning a risk score (0–100), the model's prediction, and the dataset's actual label for comparison
- Risk scores map to an action recommendation: **≥ 70** → manual review required, **≥ 30** → review recommended, below that → looks legitimate
- The top 5 features by importance are shown alongside each result, as a lightweight, feature-level explanation for the score

## Project Structure

```
credit-card-fraud-investigation/
├── app.py                    # Streamlit demo app
├── train.py                  # Training + evaluation pipeline
├── requirements.txt
├── RESULTS.md                 # Auto-generated by train.py
├── data/
│   └── demo_transactions.csv  # 40k-row sample used by the app
└── models/
    ├── fraud_model.pkl        # Trained Extra Trees classifier
    ├── scaler.pkl              # StandardScaler fit on Time/Amount
    └── feature_columns.pkl     # Feature order expected by the model
```

## Getting Started

### Installation

```bash
git clone https://github.com/snehpatel05/credit-card-fraud-investigation.git
cd credit-card-fraud-investigation
pip install -r requirements.txt
```

### Run the demo

The repo already ships a trained model, so this works immediately:

```bash
streamlit run app.py
```

### Retrain from scratch

1. Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it under `data/`
2. Update `DATA_PATH` in `train.py` to point to that file — it currently points to a local path
3. Run:

   ```bash
   python train.py
   ```

   This regenerates `models/fraud_model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`, and `RESULTS.md`

## Notes

- Class imbalance is handled via `class_weight="balanced"` rather than resampling (SMOTE, undersampling); worth comparing against as a follow-up experiment
- All five models use the scikit-learn default 0.5 decision threshold — tuning this per model against the precision-recall curve would likely close some of the gap seen in the Hist Gradient Boosting and Logistic Regression results
- `requirements.txt` doesn't pin exact versions; freezing them (`pip freeze > requirements.txt`) would make the saved model and scaler easier to reproduce reliably on another machine
