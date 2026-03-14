import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models and preprocessing tools
with open('best_class_model.pkl', 'rb') as f:
    best_class_model = pickle.load(f)
with open('best_reg_model.pkl', 'rb') as f:
    best_reg_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('le_class.pkl', 'rb') as f:
    le_class = pickle.load(f)
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('num_cols.pkl', 'rb') as f:
    num_cols = pickle.load(f)

st.title("EMIPredict AI - Financial Risk Assessment")

st.sidebar.header("Financial Profile")
age = st.sidebar.slider("Age", 18, 80, 38)
monthly_salary = st.sidebar.number_input("Monthly Salary (INR)", 0, 1000000, 50000)
monthly_rent = st.sidebar.number_input("Monthly Rent (INR)", 0, 500000, 10000)
school_fees = st.sidebar.number_input("School Fees (INR)", 0, 100000, 2000)
college_fees = st.sidebar.number_input("College Fees (INR)", 0, 100000, 0)
travel_expenses = st.sidebar.number_input("Travel Expenses (INR)", 0, 50000, 5000)
groceries_utilities = st.sidebar.number_input("Groceries & Utilities (INR)", 0, 100000, 8000)
other_monthly_expenses = st.sidebar.number_input("Other Monthly Expenses (INR)", 0, 100000, 3000)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 750)
bank_balance = st.sidebar.number_input("Bank Balance (INR)", 0, 10000000, 100000)
emergency_fund = st.sidebar.number_input("Emergency Fund (INR)", 0, 5000000, 50000)
current_emi_amount = st.sidebar.number_input("Current EMI Amount (INR)", 0, 500000, 0)

st.sidebar.header("Categorical Details")
gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education = st.sidebar.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate"])
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
company_type = st.sidebar.selectbox("Company Type", ["Private", "Government", "MNC"])
house_type = st.sidebar.selectbox("House Type", ["Rented", "Owned"])
existing_loans = st.sidebar.selectbox("Existing Loans", ["Yes", "No"])
emi_scenario = st.sidebar.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
requested_amount = st.sidebar.number_input("Requested Amount (INR)", 10000, 15000000, 100000)
requested_tenure = st.sidebar.slider("Requested Tenure (months)", 3, 240, 24)

if st.sidebar.button("Predict"):
    # Build a raw input dataframe
    input_data = {
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education': education,
        'employment_type': employment_type,
        'company_type': company_type,
        'monthly_salary': monthly_salary,
        'monthly_rent': monthly_rent,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'house_type': house_type,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'existing_loans': existing_loans,
        'emergency_fund': emergency_fund,
        'current_emi_amount': current_emi_amount,
        'emi_scenario': emi_scenario,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure
    }
    
    input_df_raw = pd.DataFrame([input_data])
    
    # Feature Engineering
    input_df_raw['total_monthly_expenses'] = (input_df_raw['school_fees'] + input_df_raw['college_fees'] + 
                                              input_df_raw['travel_expenses'] + input_df_raw['groceries_utilities'] + 
                                              input_df_raw['other_monthly_expenses'] + input_df_raw['monthly_rent'])

    # Avoid division by zero
    ms = input_df_raw['monthly_salary'].replace(0, 1)
    
    input_df_raw['dti_ratio'] = (input_df_raw['current_emi_amount'] + input_df_raw['total_monthly_expenses']) / ms
    input_df_raw['affordability_ratio'] = (ms - input_df_raw['total_monthly_expenses'] - input_df_raw['current_emi_amount']) / ms
    input_df_raw['risk_score'] = input_df_raw['credit_score'] - input_df_raw['existing_loans'].map({'Yes': 100, 'No': 0}).fillna(0)

    # Convert categoricals as we did in training
    cat_cols = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'existing_loans', 'emi_scenario']
    # Must align with feature_cols! We can just create dummy columns and then use feature_cols to align
    input_df_raw = pd.get_dummies(input_df_raw, columns=cat_cols, drop_first=False)
    
    # Create the final aligned feature vector
    final_features = pd.DataFrame(columns=feature_cols)
    final_features.loc[0] = 0 # initialize with zeros
    
    for c in feature_cols:
        if c in input_df_raw.columns:
            final_features.loc[0, c] = input_df_raw.loc[0, c]
            
    # Scale numerical columns
    final_features[num_cols] = scaler.transform(final_features[num_cols])
    
    # Predict
    eligibility_encoded = best_class_model.predict(final_features)[0]
    eligibility = le_class.inverse_transform([eligibility_encoded])[0]
    max_emi = best_reg_model.predict(final_features)[0]

    st.header("Prediction Results")
    if eligibility == 'Eligible' or eligibility == 1 or str(eligibility).lower() == 'yes':
        st.success(f"EMI Eligibility: {eligibility}")
    else:
        st.error(f"EMI Eligibility: {eligibility}")
        
    st.info(f"Max Monthly EMI recommended: ₹ {max_emi:,.2f}")
