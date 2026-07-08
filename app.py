import joblib
import pandas as pd
import streamlit as st

DATA_PATH = "data/demo_transactions.csv"
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

st.title("Smart Credit Card Fraud Investigation System")

st.write(
    "Enter a transaction row index to check whether the transaction is genuine or suspicious."
)

st.info(
    "Note: The public dataset does not contain real transaction IDs, "
    "so row index is used as a demo Transaction ID."
)

transaction_id = st.number_input(
    "Enter Transaction ID (Row Index)",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

fraud_indexes = df[df["Class"] == 1].index.tolist()

st.write("Sample fraud row indexes for demo:")
st.write(fraud_indexes[:10])

if st.button("Analyze Transaction"):
    transaction = df.iloc[[transaction_id]].copy()

    actual_class = int(transaction["Class"].iloc[0])

    X = transaction[feature_columns].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    prediction = int(model.predict(X)[0])
    fraud_probability = model.predict_proba(X)[0][1]
    risk_score = round(fraud_probability * 100, 2)

    st.subheader("Result")

    st.write("Transaction ID / Row Index:", transaction_id)
    st.write("Amount:", round(float(transaction["Amount"].iloc[0]), 2))
    st.write("Risk Score:", f"{risk_score}/100")

    if prediction == 1:
        st.error("Model Prediction: Suspicious / Fraud")
    else:
        st.success("Model Prediction: Genuine")

    st.write("Actual Class(dataset label for demo):", "Fraud" if actual_class == 1 else "Genuine")
    if risk_score >= 70:
        st.warning("Action: Manual Review Required")
    elif risk_score >= 30:
        st.warning("Action: Review Recommended")
    else:
        st.success("Action: Looks Legitimate")

    st.subheader("Transaction Details")
    st.dataframe(transaction)

    st.subheader("Top Important Features")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(5)

        st.table(importance_df)

        st.write(
            "These are the top features used by the model. "
            "Since V1 to V28 are anonymized PCA features, the explanation is feature-level."
        )
    else:
        st.write("Feature importance is not available for this model.")