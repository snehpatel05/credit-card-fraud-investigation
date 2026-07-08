# Credit Card Fraud Detection Results

## Model Comparison

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.7159 | 0.9722 |
| Decision Tree | 0.6762 | 0.7245 | 0.6995 | 0.4904 | 0.8619 |
| Random Forest | 0.8953 | 0.7857 | 0.8370 | 0.8655 | 0.9580 |
| Extra Trees | 0.9512 | 0.7959 | 0.8667 | 0.8763 | 0.9533 |
| Hist Gradient Boosting | 0.3398 | 0.8980 | 0.4930 | 0.7449 | 0.9579 |

## Best Model

Selected Model: **Extra Trees**

The best model was selected using PR-AUC as the main metric, with Recall and F1-score as secondary metrics.
