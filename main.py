import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

import mlflow
import mlflow.sklearn

def main():
    # Load cleaned CSV
    df = pd.read_csv("data/processed_data.csv", parse_dates=True)
    df['date'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('date', inplace=True)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Shift 'close' to predict next day's closing price
    df['next_close'] = df['close'].shift(-1)
    df = df.dropna()

    # Features and target
    X = df[['open', 'high', 'low', 'volume']]
    y = df['next_close']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # MLflow experiment setup
    experiment_name = "MLOps Semester Project 2025"
    mlflow.set_tracking_uri("http://localhost:5000/")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse:.4f}")

        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)
        mlflow.log_metric("mse", mse)

        input_example = X_train.iloc[[0]]
        mlflow.sklearn.log_model(model, "random_forest_model", input_example=input_example)

        joblib.dump(model, "models/stock_price_predictor.pkl")

        latest_input = df[['open', 'high', 'low', 'volume']].iloc[-1].values.reshape(1, -1)
        predicted_next = model.predict(latest_input)[0]
        print(f"Predicted next close: {predicted_next:.2f}")

# Needed for Airflow
if __name__ == "__main__":
    main()
