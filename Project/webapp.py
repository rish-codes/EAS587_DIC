import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('rf_model.pkl')

# Load the mean and standard deviation used for scaling during training
scaler = joblib.load('scaler.pkl')

def process_input(features):

    credit_score = features[0]
    age = features[3]
    tenure = features[4]
    balance = features[5]
    products_number = features[6]
    credit_card = 1 if features[7] == "Yes" else 0
    active_member = 1 if features[8] == "Yes" else 0
    est_salary = features[9]

    if features[1] == "France":
        country = [1, 0, 0]
    elif features[1] == "Germany":
        country = [0, 1, 0]
    elif features[1] == "Spain":
        country = [0, 0, 1]

    if features[2] == "Male":
        gender = [0, 1]
    elif features[2] == "Female":
        gender = [1, 0]
    
    processed_features = [credit_score, age, tenure, balance, products_number, credit_card, active_member, est_salary]
    processed_features.extend(country)
    processed_features.extend(gender)

    return processed_features

def predict(features):
    # Scale the features using the scaler
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    return prediction

def main():
    st.title("Customer Churn Prediction")

    st.write("Enter Customer Information:")
    credit_score = st.number_input("Credit Score", step=1)
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", step=1)
    tenure = st.number_input("Tenure")
    balance = st.number_input("Bank Balance")
    products_number = st.number_input("Number of products", step=1)
    credit_card = st.selectbox("Does customer own a credit card?", ["Yes", "No"])
    active_member = st.selectbox("Is customer an active member?", ["Yes", "No"])
    est_salary = st.number_input("Estimated Salary of Customer")



    # Create a button to make prediction
    if st.button("Predict"):
        # Create a list of input features
        features = [credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, est_salary]

        processed_features = process_input(features)
        # Convert the features to numpy array
        # features_array = np.array(features)

        # Make prediction
        # result = predict(features_array)

        # Determine prediction result
        # if result == 1:
        #     prediction_result = "Customer will stay."
        # else:
        #     prediction_result = "Customer will leave."

        st.write("Prediction Result:")
        st.write(processed_features)

if __name__ == "__main__":
    main()
