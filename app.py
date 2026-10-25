import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("📊 Customer Churn Intelligence Dashboard")
st.caption("AI-powered retention analytics powered by Random Forest Machine Learning Model")

st.divider()

@st.cache_resource
def load_model():
    with open('churn_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sidebar for metadata
with st.sidebar:
    st.header("📌 Model Status")
    st.success("Random Forest Classifier Active")
    st.info("Input customer parameters to evaluate retainment probability.")

# Main Form Layout
st.subheader("👤 Customer Profile Input")

col1, col2 = st.columns(2, gap="large")

with col1:
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12, help="How long the customer has been with the service")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=70.0, step=5.0)
    total_charges = tenure * monthly_charges
    st.metric(label="Calculated Lifetime Value ($)", value=f"${total_charges:,.2f}")

with col2:
    support_tickets = st.slider("Support Tickets Logged", 0, 10, 2)
    contract_type = st.selectbox(
        "Contract Type", 
        options=[0, 1, 2], 
        format_func=lambda x: {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}[x]
    )

st.divider()

# Prediction Action
if st.button("🚀 Analyze Churn Risk"):
    input_features = np.array([[tenure, monthly_charges, total_charges, support_tickets, contract_type]])
    
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🚨 **High Churn Risk Detected**")
            st.progress(probability)
            st.write(f"**Risk Score:** `{probability:.1%}`")
        else:
            st.success("✅ **Customer Retained (Low Risk)**")
            st.progress(probability)
            st.write(f"**Risk Score:** `{probability:.1%}`")
            
    with res_col2:
        st.subheader("Recommended Action")
        if prediction == 1:
            st.warning("**Retention Strategy:** Offer a 15% discount on an annual contract renewal or reach out via customer success manager.")
        else:
            st.info("**Account Status Healthy:** No immediate intervention required. Consider upselling premium features.")
