import joblib  # for loading the saved model
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load the trained model (assuming it's saved as model.pkl)
def load_model(model_path="models/stock_price_predictor.pkl"):
    model = joblib.load(model_path)
    return model


# Function to evaluate model accuracy on test data
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Example test to evaluate model accuracy
def test_model_accuracy():
    # Load a sample dataset (you would use your own test dataset)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the trained model
    model = load_model()

    # Evaluate the model on the test set
    accuracy = evaluate_model(model, X_test, y_test)

    # You can set the threshold to the minimum accuracy you expect
    expected_accuracy = 0.8  # Example threshold of 80%
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Use pytest assertions to verify if the accuracy is above the threshold
    assert accuracy >= expected_accuracy, (
        f"Model accuracy {accuracy * 100:.2f}% is below the threshold"
    )
