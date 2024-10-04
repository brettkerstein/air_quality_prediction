import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import MinMaxScaler

# Create necessary directories and get absolute paths
def setup_directories():
    # Get the absolute path to the project root (assuming process_data.py is in src/data/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create paths
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Create directories if they don't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Define file paths
    raw_data_path = os.path.join(raw_data_dir, 'openaq_data.csv')
    db_path = os.path.join(processed_data_dir, 'air_quality.db')
    
    return raw_data_path, db_path

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the raw data file at {file_path}")
        raise

def process_data(df):
    print(df.head())
    df['datetime'] =  pd.to_datetime(df['date'].str.split(':',n=1, expand=False).str[1].str.split("'",n=1, expand=False).str[1].str.split("'",n=1, expand=False).str[0])
    print(df['datetime'][0:5])
    
    df = df.sort_values('datetime').set_index('datetime')
    df_resampled = df['value'].resample('H').mean().dropna()
    return df_resampled

def store_data_sqlite(df, db_path, table_name):
    try:
        # Print debugging information
        print(f"Attempting to store data in database at: {db_path}")
        print(f"Database directory exists: {os.path.exists(os.path.dirname(db_path))}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Create connection and store data
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=True)
            print(f"Successfully stored data in {db_path}")
    
    except sqlite3.OperationalError as e:
        print(f"SQLite Error: {e}")
        print(f"Full path to database: {os.path.abspath(db_path)}")
        raise

def prepare_data_for_lstm(df, lookback):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    return X, y, scaler

if __name__ == "__main__":
    try:
        # Setup directories and get paths
        raw_data_path, db_path = setup_directories()
        
        print(f"Raw data path: {raw_data_path}")
        print(f"Database path: {db_path}")
        
        # Load the raw data
        raw_data = load_data(raw_data_path)
        print("Raw data loaded successfully")
        
        # Process the data
        processed_data = process_data(raw_data)
        print("Data processed successfully")
        
        # Store the processed data in SQLite
        store_data_sqlite(processed_data, db_path, 'measurements')
        print("Data stored in SQLite successfully")
        
        # Prepare data for LSTM (example with lookback of 24 hours)
        X, y, scaler = prepare_data_for_lstm(processed_data, lookback=24)
        print(f"Prepared {len(X)} samples for LSTM training")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise