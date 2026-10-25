import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Retention AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
    <style>
    /* Main layout adjustment */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom metric card container */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="header-box">
        <h1 style="margin:0; font-size: 28px; font-weight: 700;">⚡ Customer Churn Intelligence System</h1>
        <p style="margin-top: 8px; margin-bottom: 0; opacity: 0.8; font-size: 15px;">
            Enterprise Retention Analytics powered by Random Forest Classifier
        </p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('churn_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dashboard.png", width=64)
    st.title("Control Panel")
    st.caption("v2.1 Enterprise Edition")
    st.divider()
    
    st.markdown("### 🎯 Model Info")
    st.info("**Algorithm:** Random Forest\n\n**Accuracy:** ~88%\n\n**Status:** Production Ready")
    st.divider()
    st.caption("Built with Streamlit & Scikit-Learn")

# Input Section & Tabs
tab1, tab2 = st.tabs(["📊 Prediction Workspace", "ℹ️ Model Documentation"])

with tab1:
    st.subheader("Customer Parameters")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("**Account Attributes**")
        tenure = st.slider("Tenure Length (Months)", min_value=1, max_value=72, value=12, help="Duration of subscription in months")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=75.0, step=2.5)
        total_charges = tenure * monthly_charges
        
        st.metric(label="Calculated Customer LTV", value=f"${total_charges:,.2f}", delta=f"${monthly_charges}/mo")

    with col2:
        st.markdown("**Engagement Metrics**")
        support_tickets = st.slider("Support Tickets (Last 6 Months)", 0, 10, 1)
        contract_type = st.radio(
            "Contract Duration", 
            options=[0, 1, 2], 
            format_func=lambda x: {0: "Month-to-Month", 1: "1-Year Contract", 2: "2-Year Contract"}[x],
            horizontal=True
        )

    st.markdown("---")

    if st.button("🚀 Evaluate Churn Risk"):
        input_features = np.array([[tenure, monthly_charges, total_charges, support_tickets, contract_type]])
        
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]
        
        res_col1, res_col2 = st.columns([1, 1], gap="large")
        
        with res_col1:
            st.markdown("### 📊 Churn Probability")
            if prediction == 1:
                st.error(f"🚨 **High Risk Alert**")
                st.progress(float(probability))
                st.markdown(f"#### Risk Probability: `{probability:.1%}`")
            else:
                st.success(f"✅ **Low Risk / Retained**")
                st.progress(float(probability))
                st.markdown(f"#### Risk Probability: `{probability:.1%}`")

        with res_col2:
            st.markdown("### 💡 Recommended Action")
            if prediction == 1:
                st.warning("""
                **Priority Intervention Needed:**
                - Offer **15% discount** for switching to a 1-year contract.
                - Assign a **Dedicated Account Executive** to resolve support tickets.
                """)
            else:
                st.info("""
                **Account Healthy:**
                - Customer engagement is strong.
                - Eligible for **upselling premium features** or add-ons.
                """)

with tab2:
    st.markdown("""
    ### About the Prediction Engine
    This system predicts customer attrition risk using multi-feature pattern matching trained on customer lifecycle parameters.
    
    **Features Used:**
    - **Tenure:** Customer loyalty duration.
    - **Monthly Charges:** Current tier pricing.
    - **Total Lifetime Charges:** Total revenue generated.
    - **Support Tickets:** Customer frustration indicator.
    - **Contract Type:** Commitment lock-in level.
    """)
