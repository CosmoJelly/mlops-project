import pytest
from main import load_and_prepare_data, train_and_log_model
from sklearn.metrics import mean_squared_error

@pytest.fixture(scope="module")
def model_and_data():
    (X_train, X_test, y_train, y_test), df = load_and_prepare_data()
    model, X_test, y_test, y_pred = train_and_log_model(X_train, X_test, y_train, y_test)
    return model, X_test, y_test, y_pred

def test_model_mse(model_and_data):
    _, _, y_test, y_pred = model_and_data
    mse = mean_squared_error(y_test, y_pred)
    print(f"[TEST] MSE: {mse}")
    assert mse < 25, "Model MSE is too high – accuracy might be poor."
