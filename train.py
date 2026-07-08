import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
DATA_PATH = r"D:\Coding\credit-card-fraud-investigation\data\creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
RESULTS_PATH = "RESULTS.md"

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42,
            class_weight="balanced"
        )
    }
    return models

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("PR-AUC   :", round(pr_auc, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    result = {
        "name": name,
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }
    return result


def train_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    best_model = max(
        results,
        key=lambda x: (x["pr_auc"], x["recall"], x["f1"])
    )
    return best_model, results

def save_files(best_model, results, scaler, feature_columns):
    joblib.dump(best_model["model"], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("# Credit Card Fraud Detection Results\n\n")
        file.write("## Model Comparison\n\n")
        file.write("| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")

        for result in results:
            file.write(
                f"| {result['name']} "
                f"| {result['precision']:.4f} "
                f"| {result['recall']:.4f} "
                f"| {result['f1']:.4f} "
                f"| {result['pr_auc']:.4f} "
                f"| {result['roc_auc']:.4f} |\n"
            )

        file.write("\n## Best Model\n\n")
        file.write(f"Selected Model: **{best_model['name']}**\n\n")
        file.write(
            "The best model was selected using PR-AUC as the main metric, "
            "with Recall and F1-score as secondary metrics.\n"
        )

    print("\nFiles saved successfully:")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(FEATURE_COLUMNS_PATH)
    print(RESULTS_PATH)

def main():
    df = load_dataset()
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Fraud transactions:", df["Class"].sum())
    print("Legitimate transactions:", len(df) - df["Class"].sum())

    X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)

    best_model, results = train_models(X_train, X_test, y_train, y_test)
    save_files(best_model, results, scaler, feature_columns)
    print("\nTraining completed.")
    print("Best model:", best_model["name"])

if __name__ == "__main__":
    main()