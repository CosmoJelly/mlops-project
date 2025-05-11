import pytest
from sklearn.metrics import r2_score
from main import model, X_test, y_test

def testModel():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.4, f"Expected accuracy > 0.8 but gyot {r2:.2f}"

if __name__ == "__main__":
    pytest.main()
