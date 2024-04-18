from flask import Flask, render_template, request
import numpy as np
import joblib

# Load your trained model
model = joblib.load('rf_model.pkl')

# Load the mean and standard deviation used for scaling during training
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

def predict(features):
    # Scale the features using the scaler
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    return prediction
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    # Get input values from the form
    credit_score = int(request.form['credit_score'])
    country = request.form['country']
    gender = request.form['gender']
    age = int(request.form['age'])
    tenure = float(request.form['tenure'])
    balance = float(request.form['balance'])
    products_number = int(request.form['products_number'])
    credit_card = request.form['credit_card']
    active_member = request.form['active_member']
    est_salary = float(request.form['est_salary'])







    # Create a list of input features
    features = [[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, est_salary]]  # Add more features here if needed

    # Convert the features to numpy array
    # features_array = np.array(features)

    # Make prediction
    # result = predict(features_array)

    # Determine prediction result
    # if result == 1:
    #     prediction_result = "Customer will stay."
    # else:
    #     prediction_result = "Customer will leave."

    return render_template('result.html', prediction=features)

if __name__ == '__main__':
    app.run(debug=True)
