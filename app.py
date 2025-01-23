import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load pre-trained model and scaler
try:
    model = joblib.load('models/downtime_model.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')
except FileNotFoundError:
    from train_model import train_model
    train_model()
    model = joblib.load('models/downtime_model.pkl')
    scaler = joblib.load('models/feature_scaler.pkl')

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload manufacturing data CSV."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('data', filename)
    file.save(filepath)
    
    return jsonify({"message": f"File {filename} uploaded successfully"}), 200

@app.route('/train', methods=['POST'])
def train():
    """Train model on uploaded data."""
    from train_model import train_model
    
    # Check if data exists
    data_files = os.listdir('data')
    if not data_files:
        return jsonify({"error": "No data files found. Upload data first."}), 400
    
    # Train on most recent CSV
    latest_data = max([os.path.join('data', f) for f in data_files], key=os.path.getctime)
    
    results = train_model(latest_data)
    return jsonify(results), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions based on input features."""
    data = request.json
    
    # Validate input
    if not all(key in data for key in ['Temperature', 'Run_Time']):
        return jsonify({"error": "Missing required features"}), 400
    
    # Prepare features
    features = np.array([[data['Temperature'], data['Run_Time']]])
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)
    proba = model.predict_proba(scaled_features)
    
    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": float(proba[0][prediction[0]])
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
