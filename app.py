
import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("best_model.json")
    return model

@st.cache_resource
def load_preprocessor():
    with open("scaler_params.json", "r") as f:
        scaler_params = json.load(f)
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    return scaler_params, feature_names

model = load_model()
scaler_params, feature_names = load_preprocessor()

st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("💼 Service Details")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("🛡️ Additional Services")
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col4:
    st.subheader("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=0.5)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0,
                               float(tenure * monthly_charges), step=10.0)

st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    binary_map = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }

    for col, mapping in binary_map.items():
        input_df[col] = input_df[col].map(mapping)

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    input_df = pd.get_dummies(input_df, columns=multi_cols, drop_first=True)

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # Scale using saved parameters (same as StandardScaler)
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])
    input_scaled = (input_df.values - mean) / scale

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    churn_prob = float(probability[1] * 100)
    stay_prob = float(probability[0] * 100)

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN")
    else:
        st.success(f"✅ This customer is likely to STAY")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
    with result_col2:
        st.metric("Stay Probability", f"{stay_prob:.1f}%")

    st.markdown("**Risk Level:**")
    st.progress(float(churn_prob / 100))

    if churn_prob >= 70:
        st.warning("🔴 **HIGH RISK** — Immediate intervention recommended. "
                    "Consider offering a discount or upgraded service plan.")
    elif churn_prob >= 40:
        st.warning("🟡 **MEDIUM RISK** — Monitor this customer closely. "
                    "Proactive engagement could help retain them.")
    else:
        st.info("🟢 **LOW RISK** — Customer appears satisfied. "
                "Continue current service level.")
