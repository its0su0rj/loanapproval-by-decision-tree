import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained Decision Tree model
model = joblib.load('decisiontree.joblib')
#model = joblib.load("C:/Users/sujee/Downloads/loanapproval/decisiontree.joblib")   # Replace with the correct path to your trained model

# Streamlit UI
st.title("Loan Approval Prediction")

# Input form for user to enter data
gender = st.selectbox("Select Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.number_input("Loan Amount Term (Months)", min_value=0, step=1)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create a DataFrame with the input data
new_input = {
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area],
}

# Convert categorical variables to numeric
le = LabelEncoder()
for column in new_input.keys():
    if isinstance(new_input[column][0], str):  # Check if the value is a string
        new_input[column] = le.fit_transform(new_input[column])

# Create a DataFrame with the new input data
new_input_df = pd.DataFrame(new_input)

# Make predictions using the trained model
predicted_loan_status = model.predict(new_input_df)

# Display the predicted loan status
if st.button("Predict Loan Approval Status"):
    if predicted_loan_status[0] == 1:
        st.success("Congratulations! Your loan is approved.")
    else:
        st.error("Sorry, your loan is not approved.")
