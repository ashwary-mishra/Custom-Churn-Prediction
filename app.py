import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Custom Churn Prediction System")
st.write("Business Retention Tool using Random Forest Algorithm")

@st.cache_resource
def load_model():
    with open('churn_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=70.0)
    total_charges = tenure * monthly_charges
    st.info(f"Calculated Total Charges: ${total_charges:.2f}")

with col2:
    support_tickets = st.slider("Support Tickets Logged", 0, 10, 2)
    contract_type = st.selectbox("Contract Type", options=[0, 1, 2], 
                                 format_func=lambda x: {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}[x])

if st.button("Predict Churn Risk"):
    input_features = np.array([[tenure, monthly_charges, total_charges, support_tickets, contract_type]])
    
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"High Churn Risk! (Probability: {probability:.1%})")
        st.warning("**Recommended Action:** Offer 15% discount on 1-year contract upgrade.")
    else:
        st.success(f"Retained Customer (Churn Probability: {probability:.1%})")
        st.info("**Recommended Action:** Account is healthy. No intervention needed.")
