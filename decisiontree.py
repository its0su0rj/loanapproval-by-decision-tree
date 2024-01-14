import streamlit as st
from joblib import load
import pandas as pd

# Load the trained model
model = load("loan_approval_model.joblib")

# Define the app
def main():
    st.title("Loan Approval Predictor")

    # Input fields
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married', ['Yes', 'No'])
    Dependents = st.number_input('Dependents', min_value=0, max_value=3)
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
    ApplicantIncome = st.number_input('Applicant Income', min_value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
    LoanAmount = st.number_input('Loan Amount', min_value=0)
    Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0)
    Credit_History = st.number_input('Credit History', min_value=0, max_value=1)
    Property_Area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

    # Button to predict
    if st.button("Predict Loan Approval"):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Gender': [Gender],
            'Married': [Married],
            'Dependents': [Dependents],
            'Education': [Education],
            'Self_Employed': [Self_Employed],
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome],
            'LoanAmount': [LoanAmount],
            'Loan_Amount_Term': [Loan_Amount_Term],
            'Credit_History': [Credit_History],
            'Property_Area': [Property_Area]
        })

        # Convert categorical variables to numerical using one-hot encoding
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Ensure columns match the model's features
        missing_columns = set(model.feature_importances_) - set(input_data_encoded.columns)
        for column in missing_columns:
            input_data_encoded[column] = 0

        # Reorder columns to match the model's features
        input_data_encoded = input_data_encoded[model.feature_importances_]

        # Make predictions using the loaded model
        prediction = model.predict(input_data_encoded)

        # Display the prediction
        if prediction[0] == 'Y':
            st.write('Yes, your loan is approved')
        else:
            st.write('No, your loan is not approved')

if __name__ == '__main__':
    main()
