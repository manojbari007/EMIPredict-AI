import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import warnings
warnings.filterwarnings('ignore')

# Step 3: Data Loading and Preprocessing
def load_and_preprocess_data():
    df = pd.read_csv('emi_prediction_dataset.csv', low_memory=False)

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['monthly_salary'] = pd.to_numeric(df['monthly_salary'], errors='coerce')
    df['bank_balance'] = pd.to_numeric(df['bank_balance'], errors='coerce')

    numerical_cols = ['age', 'monthly_salary', 'monthly_rent', 'credit_score', 'bank_balance', 'emergency_fund']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    df['education'] = df['education'].fillna(df['education'].mode()[0])

    df['gender'] = df['gender'].str.upper().str.strip()
    df['gender'] = df['gender'].replace({'M': 'MALE', 'F': 'FEMALE', 'M.': 'MALE', 'F.': 'FEMALE'})

    df = df.drop_duplicates()

    return df

df = load_and_preprocess_data()
print(f"Dataset shape: {df.shape}")

# Step 5: Feature Engineering
def feature_engineering(df):
    df['total_monthly_expenses'] = (df['school_fees'] + df['college_fees'] + df['travel_expenses'] +
                                    df['groceries_utilities'] + df['other_monthly_expenses'] + df['monthly_rent'].fillna(0))

    df['dti_ratio'] = (df['current_emi_amount'] + df['total_monthly_expenses']) / df['monthly_salary']

    df['affordability_ratio'] = (df['monthly_salary'] - df['total_monthly_expenses'] - df['current_emi_amount']) / df['monthly_salary']

    df['risk_score'] = df['credit_score'] - df['existing_loans'].map({'Yes': 100, 'No': 0}).fillna(0)

    cat_cols = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'existing_loans', 'emi_scenario']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

df = feature_engineering(df)
print(f"Features after engineering: {df.shape[1]}")

le_class = LabelEncoder()
df['emi_eligibility_encoded'] = le_class.fit_transform(df['emi_eligibility'])
print("Class labels:", le_class.classes_)

# Step 6: Machine Learning Model Development
df_sample = df.sample(frac=0.1, random_state=42)
feature_cols = [col for col in df_sample.columns if col not in ['emi_eligibility', 'max_monthly_emi', 'emi_eligibility_encoded']]

X = df_sample[feature_cols]
y_class = df_sample['emi_eligibility_encoded']
y_reg = df_sample['max_monthly_emi']

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
num_cols = X.select_dtypes(include=[np.number]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

class_models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100)
}

class_results = {}
for name, model in class_models.items():
    model.fit(X_train, y_train_class)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_class, y_pred)
    class_results[name] = {'model': model, 'accuracy': acc}
    print(f"{name} Accuracy: {acc:.4f}")

best_class_name = max(class_results, key=lambda k: class_results[k]['accuracy'])
best_class_model = class_results[best_class_name]['model']
print(f"Best Classification Model: {best_class_name} with accuracy {class_results[best_class_name]['accuracy']:.4f}")

reg_models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
    'XGBoost': XGBRegressor(random_state=42, n_estimators=100)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    r2 = r2_score(y_test_reg, y_pred)
    reg_results[name] = {'model': model, 'rmse': rmse, 'r2': r2}
    print(f"{name} RMSE: {rmse:.2f}, R2: {r2:.4f}")

best_reg_name = min(reg_results, key=lambda k: reg_results[k]['rmse'])
best_reg_model = reg_results[best_reg_name]['model']
print(f"Best Regression Model: {best_reg_name} with RMSE {reg_results[best_reg_name]['rmse']:.2f}")

# Save the models and encoders
with open('best_class_model.pkl', 'wb') as f:
    pickle.dump(best_class_model, f)
with open('best_reg_model.pkl', 'wb') as f:
    pickle.dump(best_reg_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('le_class.pkl', 'wb') as f:
    pickle.dump(le_class, f)
with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(list(feature_cols), f)
with open('num_cols.pkl', 'wb') as f:
    pickle.dump(list(num_cols), f)

# Create app.py
app_code = """import streamlit as st
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
"""
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("Streamlit app code saved to app.py")
