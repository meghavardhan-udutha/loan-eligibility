import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv('credit_risk_dataset.csv')

features = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file',
    'person_income', 'person_age', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income'
]
target = 'loan_status'

X = df[features]
y = df[target]

numeric_features = [
    'person_income', 'person_age', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income'
]
categorical_features = [
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
]

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# 🚨 SAVE using joblib after installing sklearn 1.7.0!
joblib.dump(model_pipeline, 'loan_eligibility_model.pkl')

print(" Model saved using scikit-learn 1.7.0")
