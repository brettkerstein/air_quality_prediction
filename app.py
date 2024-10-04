from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = load_model('./scripts/models/lstm_model.h5', 
                   custom_objects={'MeanSquaredError': MeanSquaredError,
                                   'MSE': MSE})
scaler = joblib.load('./scripts/models/scaler.pkl')

def get_latest_data(lookback=24):
    conn = sqlite3.connect('./data/processed/air_quality.db')
    query = f"SELECT * FROM measurements ORDER BY datetime DESC LIMIT {lookback}"
    df = pd.read_sql(query, conn, index_col='datetime')
    conn.close()
    return df

def make_prediction(data):
    scaled_data = scaler.transform(data.values.reshape(-1, 1))
    X = scaled_data.reshape(1, scaled_data.shape[0], 1)
    prediction = model.predict(X)
    return scaler.inverse_transform(prediction)[0, 0]

@app.route('/api/current', methods=['GET'])
def get_current_aqi():
    latest_data = get_latest_data(lookback=1)
    print(f"lastest date: { pd.to_datetime(latest_data.index[0])}")
    return jsonify({
        'timestamp': pd.to_datetime(latest_data.index[0]).isoformat(),
        'value': float(latest_data.iloc[0]['value'])
    })

@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    data = get_latest_data(lookback=168)  # Last 7 days of hourly data
    return jsonify([
        {'timestamp': pd.to_datetime(index).isoformat(), 'value': float(row['value'])}
        for index, row in data.iterrows()
    ])

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    data = get_latest_data(lookback=24)  # Use last 24 hours for prediction
    prediction = make_prediction(data)
    return jsonify({
        'timestamp': (pd.to_datetime(data.index[-1]) + pd.Timedelta(hours=1)).isoformat(),
        'value': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)