import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_stock_data(csv_path: str) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=True)

    # Ensure date column is datetime and set as index if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    # Sort by date
    df = df.sort_index()

    # Handle missing values
    df = df.dropna()

    # Add daily return
    df['daily_return'] = df['close'].pct_change()

    # Add moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_14'] = df['close'].rolling(window=14).mean()

    # Drop initial NaNs from moving averages
    df = df.dropna()

    # Optional: Normalize features for ML
    features_to_scale = ['open', 'high', 'low', 'close', 'volume', 'ma_7', 'ma_14']
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

# Example usage
if __name__ == "__main__":
    cleaned_df = preprocess_stock_data("data/AAPL_2025-05-08.csv")
    print(cleaned_df.head())
