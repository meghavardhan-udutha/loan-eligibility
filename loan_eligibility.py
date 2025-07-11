# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Load dataset
df = pd.read_csv('streamlit-apps/loan-eligibility/credit_risk_dataset.csv')

# Step 2: Features & Target
features = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file',
    'person_income', 'person_age', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income'
]
target = 'loan_status'

X = df[features]
y = df[target]

# Step 3: Preprocessing pipelines
numeric_features = [
    'person_income', 'person_age', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income'
]
categorical_features = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
]

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Step 4: Model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Step 5: Train/test split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Step 6: Save model using joblib
joblib.dump(model_pipeline, 'loan_eligibility_model.pkl')

print(" Model retrained and saved as 'loan_eligibility_model.pkl'")
