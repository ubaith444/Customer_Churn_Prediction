"""Streamlit web application for Customer Churn Prediction."""

from pathlib import Path
import streamlit as st
from utils import load_champion_model, predict_churn

st.set_page_config(
    page_title="Telco Churn Intelligence Platform",
    layout="wide",
)

st.title("Telco Customer Churn Prediction Platform")
st.markdown(
    "Predict customer attrition risk, analyze key contributing features, and receive tailored retention recommendations."
)

# Sidebar controls
st.sidebar.header("Model and Threshold Settings")

threshold = st.sidebar.slider(
    "Decision Cutoff Threshold (tau)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Default 0.50 maximizes enterprise profit ($18,600). Lower values (0.35) increase Recall up to ~90%.",
)

st.sidebar.markdown(
    """
---
**Model Specification:**
- Architecture: Tuned XGBoost (scale_pos_weight = 2.77)
- Test ROC-AUC: `0.8453`
- Test PR-AUC: `0.6599`
- Recall @ tau=0.50: `80.75%`
"""
)

# Load model
try:
    model = load_champion_model()
    st.sidebar.success("Champion Model Loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Preset profile selection
preset = st.selectbox(
    "Load Preset Profile for Quick Demo:",
    ["Custom Input", "High-Risk Customer (Month-to-month, Fiber, New)", "Low-Risk Customer (2-Year Contract, DSL, Long Tenure)"],
)

# Initialize defaults
if preset == "High-Risk Customer (Month-to-month, Fiber, New)":
    default_gender = "Female"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 2
    default_phone = "Yes"
    default_lines = "No"
    default_internet = "Fiber optic"
    default_sec = "No"
    default_backup = "No"
    default_device = "No"
    default_tech = "No"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 95.50
    default_total = 191.00
elif preset == "Low-Risk Customer (2-Year Contract, DSL, Long Tenure)":
    default_gender = "Male"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 60
    default_phone = "Yes"
    default_lines = "Yes"
    default_internet = "DSL"
    default_sec = "Yes"
    default_backup = "Yes"
    default_device = "Yes"
    default_tech = "Yes"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Bank transfer (automatic)"
    default_monthly = 65.00
    default_total = 3900.00
else:
    default_gender = "Female"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "No"
    default_tenure = 12
    default_phone = "Yes"
    default_lines = "No"
    default_internet = "DSL"
    default_sec = "No"
    default_backup = "Yes"
    default_device = "No"
    default_tech = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 50.00
    default_total = 600.00

# Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Customer Profile")
    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
    senior = st.selectbox("Senior Citizen", [0, 1], index=default_senior, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    partner = st.selectbox("Partner", ["Yes", "No"], index=0 if default_partner == "Yes" else 1)
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if default_dependents == "Yes" else 1)
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=default_tenure)

with col2:
    st.subheader("2. Subscribed Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"], index=0 if default_lines == "No" else 1)
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(default_internet))
    security = st.selectbox("Online Security", ["No", "Yes"], index=0 if default_sec == "No" else 1)
    backup = st.selectbox("Online Backup", ["No", "Yes"], index=0 if default_backup == "No" else 1)
    device = st.selectbox("Device Protection", ["No", "Yes"], index=0 if default_device == "No" else 1)
    tech = st.selectbox("Tech Support", ["No", "Yes"], index=0 if default_tech == "No" else 1)
    tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0 if default_tv == "No" else 1)
    movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0 if default_movies == "No" else 1)

with col3:
    st.subheader("3. Account & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_contract))
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_paperless == "Yes" else 1)
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(default_payment)
    )
    monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(default_monthly), step=1.0)
    calculated_total = float(round(monthly * tenure, 2)) if tenure > 0 else 0.0
    total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total), step=10.0)

# Build customer record
customer_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

st.markdown("---")

if st.button("Evaluate Customer Churn Risk", type="primary", use_container_width=True):
    prediction, prob, risk_level, recommendations = predict_churn(model, customer_data, threshold=threshold)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Assessment Result")
        if prediction == 1:
            st.error(f"High Churn Risk (Prediction: Flagged at tau = {threshold:.2f})")
        else:
            st.success(f"Low Churn Risk (Prediction: Retained at tau = {threshold:.2f})")

        st.metric("Predicted Churn Probability", f"{prob:.1%}")
        st.progress(prob)

        if risk_level == "High Risk":
            st.markdown("**Risk Category:** High Risk (p >= 60%)")
        elif risk_level == "Moderate Risk":
            st.markdown("**Risk Category:** Moderate Risk (30% <= p < 60%)")
        else:
            st.markdown("**Risk Category:** Low Risk (p < 30%)")

    with res_col2:
        st.subheader("Recommended Retention Strategy")
        for rec in recommendations:
            st.markdown(f"- {rec}")


