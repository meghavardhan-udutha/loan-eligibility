# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('loan_eligibility_model.pkl')

# UI
st.set_page_config(page_title="Loan Eligibility Checker", layout="centered")
st.title("🏦 Loan Eligibility Prediction")
st.markdown("Fill the details to check loan approval status:")

# Inputs
person_income = st.number_input("Annual Income ($)", min_value=0)
person_age = st.number_input("Age", min_value=18, max_value=100)
person_emp_length = st.number_input("Years of Employment", min_value=0)
loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=0)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0)

person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox("Loan Purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
cb_person_default_on_file = st.selectbox("Defaulted Before?", ['Y', 'N'])

# Prediction
if st.button("Check Eligibility"):
    input_df = pd.DataFrame([{
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': cb_person_default_on_file,
        'person_income': person_income,
        'person_age': person_age,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'loan_percent_income': loan_percent_income
    }])
    
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
