import pytest
from sklearn.metrics import r2_score
from main import load_and_prepare_data, train_and_log_model

# Run the pipeline once for testing
(X_train, X_test, y_train, y_test), _ = load_and_prepare_data()
model, _, _, _ = train_and_log_model(X_train, X_test, y_train, y_test)

def test_model_r2():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.4, f"Expected R² > 0.4 but got {r2:.2f}"

if __name__ == "__main__":
    pytest.main()
