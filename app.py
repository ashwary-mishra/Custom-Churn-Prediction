import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Retention Hub",
    page_icon="🎨",
    layout="wide"
)

# Custom Vibrant & Colorful Styling
st.markdown("""
    <style>
    /* Gradient Background Header */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.25);
    }
    
    /* Vibrant Color Cards */
    .card-purple {
        background-color: #f3e8ff;
        border-left: 6px solid #9333ea;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    .card-blue {
        background-color: #e0f2fe;
        border-left: 6px solid #0284c7;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 15px;
    }

    /* Gradient Glowing Button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff007f 0%, #7928ca 100%);
        color: white;
        font-weight: 700;
        font-size: 18px;
        border-radius: 12px;
        padding: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.35);
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Main Banner
st.markdown("""
    <div class="hero-container">
        <h1 style="margin:0; font-size: 32px; font-weight: 800;">✨ Customer Retention AI Dashboard</h1>
        <p style="margin-top: 8px; font-size: 16px; opacity: 0.95;">Real-time AI churn prediction & retention insights</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('churn_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sidebar
with st.sidebar:
    st.title("🎯 Control Panel")
    st.success("🟢 Model Connected")
    st.info("💡 Tip: Adjust parameters on the right to simulate customer churn risk.")

# Input Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-purple"><h3 style="margin:0; color:#581c87;">👤 Customer Account Details</h3></div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure Duration (Months)", min_value=1, max_value=72, value=18)
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=100.0, max_value=10000.0, value=1500.0, step=100.0)
    total_charges = tenure * monthly_charges
    st.metric(label="Calculated Lifetime Value (₹)", value=f"₹{total_charges:,.2f}", delta="Active")

with col2:
    st.markdown('<div class="card-blue"><h3 style="margin:0; color:#0369a1;">📊 Service Usage & Support</h3></div>', unsafe_allow_html=True)
    support_tickets = st.slider("Support Tickets Raised", 0, 10, 2)
    contract_type = st.radio(
        "Select Contract Type", 
        options=[0, 1, 2], 
        format_func=lambda x: {0: "Month-to-Month 📄", 1: "1-Year Contract 📜", 2: "2-Year Contract 🔐"}[x],
        horizontal=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("🚀 Analyze Churn Risk"):
    input_features = np.array([[tenure, monthly_charges, total_charges, support_tickets, contract_type]])
    
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]
    
    st.markdown("---")
    
    res1, res2 = st.columns(2)
    
    with res1:
        st.subheader("🎯 Risk Assessment")
        if prediction == 1:
            st.error(f"🔴 **High Churn Risk Detected!**")
            st.progress(float(probability))
            st.write(f"**Risk Score:** `{probability:.1%}`")
        else:
            st.success(f"🟢 **Customer Retained (Low Risk)**")
            st.progress(float(probability))
            st.write(f"**Risk Score:** `{probability:.1%}`")
            
    with res2:
        st.subheader("💡 Recommended Action Plan")
        if prediction == 1:
            st.warning("⚠️ **Immediate Action:** Offer an upgrade discount on annual contract to reduce churn risk.")
        else:
            st.info("🎉 **Healthy Account:** Customer is engaged. Offer loyalty rewards or premium upgrades.")
