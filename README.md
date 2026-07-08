# Credit Card Fraud Investigation System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## Project Links

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://credit-card-fraud-investigation.streamlit.app/)

[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)]([https://youtu.be/wfixrbYbE8g])

---

## Overview

This project is a machine learning based fraud investigation system built on the ULB Credit Card Fraud Detection dataset. The goal is to identify suspicious credit card transactions and present the result in a simple investigator-facing Streamlit app.

Fraud detection is a rare-event problem. In the full dataset, only 492 out of 284,807 transactions are fraudulent, which is around 0.172%. Because of this imbalance, accuracy alone is not useful. A model that predicts every transaction as genuine would still look highly accurate, but it would fail completely at detecting fraud.

This project focuses on fraud-aware evaluation using precision, recall, F1-score, PR-AUC, and ROC-AUC.

---

## Dataset

Dataset used:

[Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains credit card transactions made by European cardholders in September 2013.

Main columns:

- `Time`: seconds elapsed from the first transaction in the dataset
- `Amount`: transaction amount
- `V1` to `V28`: anonymized PCA-transformed features
- `Class`: target column

Target values:

- `0` = genuine transaction
- `1` = fraudulent transaction

The full dataset contains:

```text
Total transactions: 284,807
Fraud transactions: 492
Genuine transactions: 284,315
```

---

## Important Deployment Note

The full `creditcard.csv` file is not committed to this GitHub repository because it is too large.

The model was trained locally using the full Kaggle dataset. For deployment, the Streamlit app uses:

```text
data/demo_transactions.csv
```

This file contains the first 40,000 rows from the original dataset. Since the rows are copied in the same order:

```text
demo row 0     = original dataset row 0
demo row 1     = original dataset row 1
demo row 39999 = original dataset row 39999
```

So, for this deployed demo, the row index entered in the app matches the original dataset row index from `0` to `39999`.

---

## Project Workflow

```text
Full Kaggle dataset
        |
        v
Preprocess features
        |
        v
Train multiple ML models
        |
        v
Evaluate using fraud-focused metrics
        |
        v
Select best model
        |
        v
Save model, scaler, and feature columns
        |
        v
Use Streamlit app for transaction-level investigation
```

---

## Models Trained

The training pipeline compares multiple machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Hist Gradient Boosting

The best model is selected using:

```text
Primary metric   : PR-AUC
Secondary metric : Recall
Supporting metric: F1-score
```

PR-AUC is used as the main metric because fraud detection is highly imbalanced and the fraud class is very rare.

---

## Results

Model comparison from `RESULTS.md`:

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.7159 | 0.9722 |
| Decision Tree | 0.6762 | 0.7245 | 0.6995 | 0.4904 | 0.8619 |
| Random Forest | 0.8953 | 0.7857 | 0.8370 | 0.8655 | 0.9580 |
| **Extra Trees** | **0.9512** | 0.7959 | **0.8667** | **0.8763** | 0.9533 |
| Hist Gradient Boosting | 0.3398 | 0.8980 | 0.4930 | 0.7449 | 0.9579 |

The selected model is **Extra Trees** because it achieved the best PR-AUC and strong precision, while still maintaining useful recall.

---

## Feature Importance

The final model provides model-level feature importance.

Top important features:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | V14 | 0.1350 |
| 2 | V4 | 0.1215 |
| 3 | V17 | 0.1109 |
| 4 | V12 | 0.1014 |
| 5 | V11 | 0.0847 |

Since `V1` to `V28` are anonymized PCA features, their exact business meaning is not available. The explanation is therefore shown at feature level.

A future version can use SHAP or LIME for stronger per-transaction explanations.

---

## Streamlit App

The Streamlit app allows a user to:

- enter a transaction row index
- get a fraud/genuine prediction
- view a fraud risk score from 0 to 100
- compare the prediction with the actual dataset label for demo purposes
- view transaction details
- view model-level important features

The actual class is shown only because this is a labeled public dataset. In a real banking system, the true label would usually be confirmed later after investigation.

---

## Risk Score Meaning

The model returns a fraud probability, which is converted into a risk score:

```text
Risk Score = Fraud Probability x 100
```

Prototype action bands:

```text
Risk score >= 70 : Manual Review Required
Risk score >= 30 : Review Recommended
Risk score < 30  : Looks Legitimate
```

These thresholds are simple prototype rules and can be tuned further based on business requirements.

---

## Project Structure

```text
credit-card-fraud-investigation/
│
├── app.py
├── train.py
├── README.md
├── RESULTS.md
├── requirements.txt
│
├── data/
│   └── demo_transactions.csv
│
└── models/
    ├── fraud_model.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
```

---

## Saved Files

After training, the following files are saved:

```text
models/fraud_model.pkl
models/scaler.pkl
models/feature_columns.pkl
```

Purpose:

- `fraud_model.pkl`: trained fraud detection model
- `scaler.pkl`: scaler fitted on `Time` and `Amount`
- `feature_columns.pkl`: exact feature order expected by the model

These files allow the app to make predictions without retraining.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/snehpatel05/credit-card-fraud-investigation.git
cd credit-card-fraud-investigation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Retrain From Scratch

To retrain the model locally:

1. Download `creditcard.csv` from the Kaggle dataset page.
2. Place it inside the `data/` folder.
3. Make sure `train.py` points to the full dataset.
4. Run:

```bash
python train.py
```

This will regenerate:

```text
models/fraud_model.pkl
models/scaler.pkl
models/feature_columns.pkl
RESULTS.md
```

---

## Evaluation Metrics

The project uses fraud-focused evaluation metrics:

- **Precision**: out of transactions predicted as fraud, how many were actually fraud
- **Recall**: out of actual fraud transactions, how many were caught
- **F1-score**: balance between precision and recall
- **PR-AUC**: precision-recall performance, useful for imbalanced datasets
- **ROC-AUC**: overall ability to separate fraud and genuine transactions

Accuracy is not used as the main metric because the dataset is highly imbalanced.

---

## Limitations

- The dataset is anonymized, so `V1` to `V28` cannot be explained in normal business language.
- The deployed app uses the first 40,000 rows because the full dataset is too large for GitHub upload.
- The current app shows model-level feature importance, not full SHAP/LIME transaction-level explanation.
- Thresholds for manual review are prototype rules.
- In a real system, the model should be retrained regularly on newer transaction data.

---

## Future Improvements

- Add SHAP or LIME for per-transaction explanations
- Add time-based validation for more realistic fraud monitoring
- Add investigator filters for high-risk transactions
- Add case status and investigator notes
- Store the full dataset using cloud storage or Git LFS
- Tune decision thresholds using the precision-recall curve

---

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## Summary

This project demonstrates a complete fraud investigation prototype. It trains multiple machine learning models on the full credit card fraud dataset, selects the best-performing model using fraud-aware metrics, saves the trained model, and provides a simple Streamlit interface for transaction-level fraud risk analysis.
