import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("models/fraud_xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

st.subheader("Transaction Details")

# Create input fields
transaction = {}

for feature in [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]:
    transaction[feature] = st.number_input(
        feature,
        value=0.0
    )

if st.button("Predict Transaction"):

    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Get fraud probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Apply threshold
    prediction = probability >= threshold

    st.subheader("Prediction")

    if prediction:
        st.error("🚨 FRAUDULENT TRANSACTION")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")

    st.write(f"Fraud Probability: **{probability:.2%}**")
    st.write(f"Decision Threshold: **{threshold:.2f}**")