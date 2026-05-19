# =========================================
# STREAMLIT CHURN PREDICTION APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================================
# ADVANCED UI + ANIMATIONS
# =========================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}

/* BACKGROUND ANIMATION */
@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* TITLE ANIMATION */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: white;
    animation: fadeInDown 1s ease-in-out;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #d1d5db;
    margin-bottom: 30px;
    animation: fadeIn 2s ease-in-out;
}

/* CARD DESIGN */
.css-1r6slb0, .st-emotion-cache-1r6slb0 {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(17,24,39,0.95);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.4s;
    animation: pulse 2s infinite;
}

.stButton>button:hover {
    transform: scale(1.04);
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    box-shadow: 0px 0px 20px rgba(59,130,246,0.7);
}

/* INPUT BOX */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    border-radius: 12px !important;
}

/* ANIMATIONS */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* BUTTON PULSE */
@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(59,130,246,0.7);
    }
    70% {
        box-shadow: 0 0 0 12px rgba(59,130,246,0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(59,130,246,0);
    }
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    transition: 0.4s;
    animation: fadeIn 1.5s ease-in-out;
}

.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0px 0px 20px rgba(255,255,255,0.1);
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 20px;
    color: #9ca3af;
    font-size: 15px;
    animation: fadeIn 3s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown('<div class="title">📊 Telecom Customer Churn Prediction</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI Powered System to Predict Whether a Customer Will Leave or Stay</div>',
    unsafe_allow_html=True
)

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():

    df = pd.read_csv("Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    df.drop(columns=["customerID"], inplace=True)

    le = LabelEncoder()

    df["Churn"] = le.fit_transform(df["Churn"])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# =========================================
# TRAIN MODEL
# =========================================
x = df.drop("Churn", axis=1)
y = df["Churn"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(x, y)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("🧾 Customer Details")

st.sidebar.markdown("Fill all customer information below.")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["No", "DSL", "Fiber optic"]
)

online_sec = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    value=12
)

# =========================================
# DASHBOARD METRICS
# =========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{df.shape[0]}</h2>
        <p>Total Customers</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{df.shape[1]-1}</h2>
        <p>Total Features</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h2>Random Forest</h2>
        <p>ML Model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# ENCODING
# =========================================
def encode_input():

    return np.array([[
        0 if gender == "Female" else 1,

        1 if senior == "Yes" else 0,

        1 if partner == "Yes" else 0,

        1 if dependents == "Yes" else 0,

        1 if phone == "Yes" else 0,

        0 if multiple == "No" else 1,

        0 if internet == "No"
        else (1 if internet == "DSL" else 2),

        1 if online_sec == "Yes" else 0,

        1 if online_backup == "Yes" else 0,

        1 if device == "Yes" else 0,

        1 if tech == "Yes" else 0,

        1 if tv == "Yes" else 0,

        1 if movies == "Yes" else 0,

        0 if contract == "Month-to-month"
        else (1 if contract == "One year" else 2),

        1 if paperless == "Yes" else 0,

        0 if payment == "Electronic check"
        else 1 if payment == "Mailed check"
        else 2 if payment == "Bank transfer (automatic)"
        else 3,

        monthly,
        total,
        tenure
    ]])

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Predict Customer Churn"):

    input_data = encode_input()

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:

        st.error(
            f"⚠ Customer is likely to CHURN\n\nRisk Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Customer is NOT likely to churn\n\nRetention Probability: {(1-probability):.2%}"
        )

# =========================================
# DATA PREVIEW SECTION
# =========================================
st.markdown("## 📄 Dataset Preview")

st.dataframe(
    df.head(10),
    width="stretch"
)

# =========================================
# FOOTER
# =========================================
st.markdown("""
<div class="footer">
     Built by Ayan Baig | Machine Learning Customer Churn Prediction System | © 2026
</div>
""", unsafe_allow_html=True)