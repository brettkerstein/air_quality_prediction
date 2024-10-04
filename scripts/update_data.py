import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv, find_dotenv

def fetch_openaq_data(city, parameter, date_from, date_to, api_key):
    url = "https://api.openaq.org/v2/measurements"
    headers = {
        "X-API-Key": api_key
    }
    params = {
        "city": city,
        "parameter": parameter,
        "date_from": date_from,
        "date_to": date_to,
        "limit": 10000
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data['results'])

def process_data(df):
    df['datetime'] = pd.to_datetime(df['date.utc'])
    df = df.sort_values('datetime').set_index('datetime')
    df_resampled = df['value'].resample('H').mean().dropna()
    return df_resampled

def update_sqlite_data(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    
    # Get the latest date in the existing data
    try:
        latest_date = pd.read_sql(f"SELECT MAX(datetime) FROM {table_name}", conn).iloc[0, 0]
        latest_date = pd.to_datetime(latest_date)
    except:
        latest_date = None
    
    # Filter new data to only include dates after the latest existing date
    if latest_date:
        df = df[df.index > latest_date]
    
    # Append new data to the existing table
    df.to_sql(table_name, conn, if_exists='append', index=True)
    conn.close()

if __name__ == "__main__":
    # Get API key from environment variable
    load_dotenv('.env')
    
    api_key = os.environ.get("OPENAQ_API_KEY")
    print(f"API: {api_key}")
    
    if not api_key:
        raise ValueError("OPENAQ_API_KEY environment variable is not set")

    city = "Los Angeles"
    parameter = "pm25"
    date_to = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Fetch last 7 days of data
    
    # Fetch new data
    df = fetch_openaq_data(city, parameter, date_from, date_to, api_key)
    
    # Process the data
    processed_data = process_data(df)
    
    # Update the SQLite database
    update_sqlite_data(processed_data, 'data/processed/air_quality.db', 'measurements')
    
    print(f"Data updated in data/processed/air_quality.db")