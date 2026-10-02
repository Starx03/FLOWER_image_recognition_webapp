import streamlit as st
import pandas as pd
import joblib

# 1. PAGE SETUP
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Eligibility Predictor")
st.write("Enter applicant details below to check loan approval status in real-time.")

# 2. LOAD THE TRAINED MODEL
@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Error loading 'loan_model.pkl'. Make sure you ran train_loan_model.py first!")
    st.stop()

# 3. CREATE INPUT FORM FOR USER DATA
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
    income_annum = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)

with col2:
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=150000, step=5000)
    loan_term = st.slider("Loan Term (Years)", min_value=1, max_value=30, value=10)
    cibil_score = st.slider("CIBIL / Credit Score", min_value=300, max_value=900, value=750)
    
st.subheader("Assets Value")
col3, col4 = st.columns(2)

with col3:
    residential_assets_value = st.number_input("Residential Assets ($)", min_value=0, value=100000, step=5000)
    commercial_assets_value = st.number_input("Commercial Assets ($)", min_value=0, value=0, step=5000)

with col4:
    luxury_assets_value = st.number_input("Luxury Assets ($)", min_value=0, value=10000, step=1000)
    bank_asset_value = st.number_input("Bank Asset Value ($)", min_value=0, value=25000, step=1000)

# Convert string inputs to numerical format matching training data
education_num = 1 if education == "Graduate" else 0
self_employed_num = 1 if self_employed == "Yes" else 0

# 4. MAKE PREDICTION ON BUTTON CLICK
if st.button("Predict Loan Status", type="primary"):
    # Construct input dataframe matching exact feature names
    input_data = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'education': education_num,
        'self_employed': self_employed_num,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }])
    
    # Run model prediction
    prediction = model.predict(input_data)[0]
    
    st.divider()
    if prediction == 1:
        st.success("**Loan Approved!** The applicant meets the criteria.")
    else:
        st.error("**Loan Rejected.** The applicant does not meet the necessary threshold.")