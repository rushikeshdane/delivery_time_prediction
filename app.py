from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import warnings


# Load the trained model
model = joblib.load("delivery_time_model.pkl")

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        distance = float(request.form['distance'])
        weather = request.form['weather']
        traffic = request.form['traffic']
        time_of_day = request.form['time_of_day']
        vehicle = request.form['vehicle']
        prep_time = int(request.form['prep_time'])
        experience = float(request.form['experience'])

        # Prepare input data for the model
        input_data = np.array([[distance, prep_time, experience, weather, traffic, time_of_day, vehicle]])
        column_names = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
                        "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
        input_df = pd.DataFrame(input_data, columns=column_names)

        # Make a prediction
        prediction = model.predict(input_df)
        predicted_time = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Delivery Time: {predicted_time} minutes')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
