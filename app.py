from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import traceback
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data
def load_data(file):
    try:
        data = pd.read_csv(file)
        app.logger.debug(f'Data loaded successfully with shape: {data.shape}')
        
        # Convert date column to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            app.logger.debug('Date column converted to datetime.')
        
        return data
    except Exception as e:
        app.logger.error(f'Error loading data: {str(e)}')
        raise

# Train model
def train_model(X, y, model_type):
    try:
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            model = XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            model = LGBMRegressor(n_estimators=100, random_state=42)
        elif model_type == 'catboost':
            model = CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
        else:
            raise ValueError("Invalid model type")
        
        model.fit(X, y)
        app.logger.debug(f'Model trained successfully: {model_type}')
        return model
    except Exception as e:
        app.logger.error(f'Error training model: {str(e)}')
        raise

# Global variable to store the last known data point
last_known_data = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global last_known_data  # Use the global variable
    try:
        model_type = request.form.get('model')
        if not model_type:
            return jsonify({'error': 'Model type not specified'}), 400

        if 'data' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['data']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Load data
        data = load_data(file)
        
        # Assuming the last column is the target variable and the first column is the date
        X = data.iloc[:, 1:-1]  # Exclude the date column and the target column
        y = data.iloc[:, -1]
        last_known_data = X.iloc[-1].tolist()  # Store the last known data point
        app.logger.debug(f'Last known data point: {last_known_data}')
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'scaler.joblib')
        app.logger.debug('Data scaled and scaler saved.')
        
        # Train model
        model = train_model(X_scaled, y, model_type)
        
        # Save the model
        joblib.dump(model, 'model.joblib')
        app.logger.debug('Model saved successfully.')
        
        return jsonify({'message': f'{model_type.capitalize()} model trained successfully'})
    except Exception as e:
        app.logger.error(f'An error occurred: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global last_known_data  # Use the global variable
    try:
        data = request.get_json()
        app.logger.debug(f'Received request data: {data}')

        if not data or 'model_type' not in data:
            return jsonify({'error': 'Invalid or missing model_type'}), 400

        model_type = data['model_type']

        # Ensure that the model files exist
        if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400

        # Ensure that last_known_data is set
        if last_known_data is None:
            return jsonify({'error': 'No data available for prediction. Please train the model first.'}), 400

        app.logger.debug(f'last_known_data: {last_known_data}')

        # Load the saved model and scaler
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')

        # Prepare for multiple predictions
        predictions = []
        input_data = last_known_data.copy()  # Use the last known data point
        for _ in range(30):  # Predict for 30 days
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]
            predictions.append(float(prediction))  # Convert to float for JSON serialization
            # Update input_data for the next prediction
            input_data = input_data[1:] + [prediction]  # Shift the input data

        return jsonify({'predictions': predictions})
    except Exception as e:
        app.logger.error(f'An error occurred: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
