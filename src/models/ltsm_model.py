import numpy as np
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_data_sqlite(db_name, table_name):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn, index_col='datetime')
    conn.close()
    return df

def prepare_data_for_lstm(df, lookback):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

if __name__ == "__main__":
    # Load data from SQLite
    df = load_data_sqlite('data/processed/air_quality.db', 'measurements')
    
    # Prepare data for LSTM
    lookback = 24  # Use 24 hours of historical data to predict the next hour
    X, y, scaler = prepare_data_for_lstm(df, lookback)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train LSTM model
    model = create_lstm_model((lookback, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)
    
    # Print some results
    print("Actual vs Predicted:")
    for i in range(10):
        print(f"Actual: {actual[i][0]:.2f}, Predicted: {predictions[i][0]:.2f}")
    
    # Save the model
    model.save('models/lstm_model.h5')
    print("Model saved to models/lstm_model.h5")