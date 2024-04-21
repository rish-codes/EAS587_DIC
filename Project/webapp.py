import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px

import os
os.chdir('Project')

df = pd.read_csv("webapp_df.csv")
df['active_member'] = df['active_member'].replace({1: 'Active Member', 0: 'Inactive Member'})

# Load your trained model
model = joblib.load('svc_model.pkl')

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

    scaled_feature_indices = [0, 1, 2, 3, 7]
    features[:, scaled_feature_indices] = scaler.transform(features[:, scaled_feature_indices])

    # Make prediction
    prediction = model.predict(features)

    return prediction

def main():

    st.set_page_config(layout="wide", page_title="Customer Churn Prediction 📈", page_icon="📈")
    st.title("Customer Churn Prediction 📈")

    # Sidebar
    st.sidebar.title(":red[Overview]")
    st.sidebar.write("Customer Churn Prediction allows banks to predict if a customer would stay or leave the bank based on some metrics and characteristcs.")
    st.sidebar.subheader(":red[Churn Prediction]")
    st.sidebar.write("Input customer details like age, gender, income, country of residence, etc. to predict if the customer would stay with the bank or leave.")
    st.sidebar.subheader(":red[Customer Statistics]")
    st.sidebar.write("Select a category to view visuals of the distribution of customers based on that category.")
    st.sidebar.subheader(":red[Churn Statistics]")
    st.sidebar.write("Select a category to view visuals of the current churn of customers by that category.")
    st.sidebar.subheader(":red[Dataset Viewer]")
    st.sidebar.write("View the raw dataset for detailed information.")
    

    tab1, tab2, tab3, tab4 = st.tabs(["Predict Churn", "Customer Statistics", "Churn Statistics", "View Dataset"])


    with tab1:

# TODO: fake progress bar lmao 

        st.write("Enter Customer Information:")

        col1, col2 = st.columns([1, 1])

        with col1:
            credit_score = st.number_input("Credit Score", step=1)
            country = st.selectbox("Country", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", step=1)
            tenure = st.number_input("Tenure")

        with col2:
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
            processed_features = np.array(processed_features).reshape(1, -1)

            # Make prediction
            result = predict(processed_features)

            # Determine prediction result
            if result == 1:
                prediction_result = "Customer will leave the bank."
            else:
                prediction_result = "Customer will stay."

            st.subheader("Prediction Result:")
            st.markdown(f"<h2 style='text-align: center; color: {'#4CAF50' if result == 0 else '#FF4B4B'};'>{prediction_result}</h2>", unsafe_allow_html=True)

    with tab2:
        st.subheader("View customer distributions by category")

        categories = ["Age", "Tenure", "Credit Score", "Balance", "Number of Products Enrolled", "Income"]
        selected_category = st.selectbox("Select Category", categories)

        if selected_category == "Age":
            fig = px.histogram(df, x='age', nbins=20, title="Age Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)
        
        elif selected_category == "Tenure":
            fig = px.histogram(df, x='tenure', nbins=20, title="Tenure Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Credit Score":
            fig = px.histogram(df, x='credit_score', nbins=20, title="Credit Score Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Balance":
            fig = px.histogram(df, x='balance', nbins=20, title="Balance Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Number of Products Enrolled":
            fig = px.histogram(df, x='products_number', nbins=20, title="Number of Products Enrolled Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Income":
            fig = px.histogram(df, x='estimated_salary', nbins=20, title="Income Distribution", color_discrete_sequence=['#FF4B4B'])
            fig.update_traces(marker_line_width=3, marker_line_color=["#262730"])
            st.plotly_chart(fig, use_container_width=False)


    with tab3:
        st.subheader("Churn Statistics")

        categories = ["Gender", "Country", "Credit Score Standing", "Active Member Status"]
        selected_category = st.selectbox("Select Category", categories)

        if selected_category == "Gender":
            fig = px.pie(df, names='gender', values='churn', title='Churn Distribution by Gender', hole=0.3, color_discrete_sequence=['#FF4B4B', '#262730'])
            st.plotly_chart(fig, use_container_width=False)
        
        elif selected_category == "Country":
            fig = px.pie(df, names='country', values='churn', title='Churn Distribution by Country', hole=0.3, color_discrete_sequence=['#FF4B4B', '#262730', '#AA3232'])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Credit Score Standing":
            fig = px.pie(df, names='Credit_Category', values='churn', title='Churn Distribution by Credit Score Standing', hole=0.3, color_discrete_sequence=['#FF4B4B', '#262730', '#AA3232'])
            st.plotly_chart(fig, use_container_width=False)

        elif selected_category == "Active Member Status":
            fig = px.pie(df, names='active_member', values='churn', title='Churn Distribution by Active Member Status', hole=0.3, color_discrete_sequence=['#FF4B4B', '#262730'])
            st.plotly_chart(fig, use_container_width=False)


    with tab4:
        st.subheader("Customer Dataset")
        st.dataframe(df)


if __name__ == "__main__":
    main()
