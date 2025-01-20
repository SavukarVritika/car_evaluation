from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('car_evaluation_model(2).model')
le = joblib.load('label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    form_data = {
        'buying_price': 'vhigh',
        'maintenance_price': 'vhigh',
        'doors': '2',
        'capacity': '2',
        'luggage_boot': 'small',
        'safety': 'low'
    }
    
    if request.method == 'POST':
        try:
            # Update form_data with submitted values
            form_data = {
                'buying_price': request.form['buying_price'],
                'maintenance_price': request.form['maintenance_price'],
                'doors': request.form['doors'],
                'capacity': request.form['capacity'],
                'luggage_boot': request.form['luggage_boot'],
                'safety': request.form['safety']
            }
            
            # Prepare input data for prediction
            input_data = {
                'Buying price': form_data['buying_price'],
                'maintenance price': form_data['maintenance_price'],
                'doors': form_data['doors'],
                'capacity': form_data['capacity'],
                'luggage boot': form_data['luggage_boot'],
                'safety': form_data['safety']
            }
            
            # Transform input data using the same label encoders
            encoded_input = {}
            for key, value in input_data.items():
                encoded_input[key] = le[key].transform([value])[0]
            
            # Create input array for prediction
            input_array = np.array([list(encoded_input.values())])
            
            # Make prediction
            pred_encoded = model.predict(input_array)[0]
            prediction = le['evaluation level'].inverse_transform([pred_encoded])[0]
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
