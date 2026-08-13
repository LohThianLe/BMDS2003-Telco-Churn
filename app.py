import streamlit as st
import pandas as pd
import joblib
import os

st.title("Telco Customer Churn Predictor")
st.write("Enter a customer's details to predict whether they are likely to churn.")

# ---- Model selection ----
# Note: Logistic Regression excluded for now — it needs scaled input,
# but the scaler used in preprocessing.py was never saved to a file.
model_options = {
    "Random Forest": "outputs/random_forest_model.pkl",
    "Decision Tree": "outputs/decision_tree_model.pkl",
    "XGBoost": "outputs/xgboost_model.pkl",
    "Logistic Regression": "outputs/logistic_regression_model.pkl",
}

selected_model_name = st.selectbox("Choose a model", list(model_options.keys()))
model_path = model_options[selected_model_name]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Make sure it's been trained and saved.")
    st.stop()

model = joblib.load(model_path)

# ---- Input form ----
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])

with col2:
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ])

# ---- Predict button ----
if st.button("Predict Churn"):

    input_dict = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 1 if multiple_lines == "Yes" else 0,
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "OnlineBackup": 1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies": 1 if streaming_movies == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    if selected_model_name == "Logistic Regression":
        scaler = joblib.load("outputs/scaler.pkl")
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Result")
    st.caption(f"Model used: {selected_model_name}")
    if prediction == 1:
        st.error(f"This customer is likely to CHURN. (Probability: {probability:.1%})")
    else:
        st.success(f"This customer is likely to STAY. (Probability of churn: {probability:.1%})")