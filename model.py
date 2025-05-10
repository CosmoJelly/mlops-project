import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

import mlflow
import mlflow.sklearn


# Load cleaned CSV
df = pd.read_csv("data/processed_data.csv", parse_dates=True)
df['date'] = pd.to_datetime(df['Unnamed: 0'])  # restore date if index was saved as column
df.set_index('date', inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Shift 'close' to predict next day's closing price
df['next_close'] = df['close'].shift(-1)

# Drop last row (has no target value)
df = df.dropna()

# Features and target
X = df[['open', 'high', 'low', 'volume']]
y = df['next_close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


experiment_name = "MLOps Semester Project 2025"
mlflow.set_tracking_uri("http://localhost:5000/")

# Try to get the experiment by name
experiment = mlflow.get_experiment_by_name(experiment_name)

# If it exists, use its ID; otherwise, create it
if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(experiment_name)

with mlflow.start_run(experiment_id=experiment_id):
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")

    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("random_state", model.random_state)
    mlflow.log_metric("mse", mse)

    input_example = X_train.iloc[[0]]

    mlflow.sklearn.log_model(model,
                             "random forest model",
                             input_example=input_example)

    # Save model
    joblib.dump(model, "models/stock_price_predictor.pkl")

    # Predict next closing price based on latest row
    latest_input = df[['open', 'high', 'low', 'volume']].iloc[-1].values.reshape(1, -1)
    predicted_next = model.predict(latest_input)[0]
    print(f"Predicted next close: {predicted_next:.2f}")
