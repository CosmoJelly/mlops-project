import requests
import pandas as pd
import os

API_KEY = "P2B3G6LGD9G5A2P9"  # Replace this with your actual key
SYMBOL = "AAPL"  # Stock symbol to track (e.g., Apple)
OUTPUT_DIR = "data"  # Folder to save CSV
API_URL = "https://www.alphavantage.co/query"


# === FUNCTION TO FETCH DATA ===
def fetch_stock_data(symbol, api_key):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact",  # use 'full' for full history
        "datatype": "json"
    }
    print("[INFO] Fetching data...")
    response = requests.get(API_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"API error: {data}")

    ts_data = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)
    return df


# === MAIN ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        df = fetch_stock_data(SYMBOL, API_KEY)
        df.to_csv(
            os.path.join(OUTPUT_DIR, "raw_data.csv")
        )
        print("[SUCCESS] Data saved")
    except Exception as e:
        print(f"[ERROR] Failed to fetch or save data: {e}")
