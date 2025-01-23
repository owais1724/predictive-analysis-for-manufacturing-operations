import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic manufacturing data."""
    np.random.seed(42)
    
    # Generate features
    machine_id = np.random.randint(1, 11, n_samples)
    temperature = np.random.uniform(50, 120, n_samples)
    run_time = np.random.uniform(30, 240, n_samples)
    
    # Create downtime flag based on synthetic rules
    downtime_flag = (
        (temperature > 100) | 
        (run_time > 180) | 
        (machine_id % 3 == 0)
    ).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Machine_ID': machine_id,
        'Temperature': temperature,
        'Run_Time': run_time,
        'Downtime_Flag': downtime_flag
    })
    
    return data

def train_model(data_path=None):
    """Train a machine downtime prediction model."""
    # Use synthetic data if no path provided
    if data_path is None:
        data = generate_synthetic_data()
    else:
        data = pd.read_csv(data_path)
    
    # Prepare features and target
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save model and scaler
    joblib.dump(model, 'models/downtime_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred)
    }

if __name__ == '__main__':
    results = train_model()
    print("Model Training Results:")
    print(f"Accuracy: {results['accuracy']}")
    print(f"F1 Score: {results['f1_score']}")
    print("\nClassification Report:")
    print(results['classification_report'])

