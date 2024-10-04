import requests
import pandas as pd
from datetime import datetime, timedelta
import os
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
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    return pd.DataFrame(data['results'])

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
    date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    df = fetch_openaq_data(city, parameter, date_from, date_to, api_key)
    print(df.head())
    # Save the fetched data to a CSV file
    df.to_csv("openaq_data.csv", index=False)
    df.to_csv("../../data/raw/openaq_data.csv", index=False)
    print(f"Data fetched and saved to data/raw/openaq_data.csv")
