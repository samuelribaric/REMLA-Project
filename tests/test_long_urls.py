"""Utility module for testing model on the test dataset"""
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test

def test_model():
    """Tests the saved model on long URLs in the test dataset."""
    np.random.seed(42)  # Set seed for reproducibility
    x_test, y_test = load_test_data()

    # Print the total number of datapoints initially
    print("Total number of datapoints:", len(x_test))

    # Define the minimum number of non-zero tokens for a URL to be considered long
    min_tokens = 100

    # Note: A token consists of 200 digits.
    # If a URL has X < 200 characters,
    # in its tokenized form, it will have X non-zero entries.
    # Thus, the above condition is equivalent to saying
    # that a URL has to have more than 100 characters.

    # Filter the test data to include only sequences with more than `min_tokens` non-zero entries
    long_urls_indices = [i for i, sequence in enumerate(x_test) if np.count_nonzero(sequence) > min_tokens]
    x_test_long = x_test[long_urls_indices]
    y_test_long = y_test[long_urls_indices]

    # Print the number of long datapoints
    print("Number of long datapoints:", len(x_test_long))

    model = load_model("models/model.keras")

    loss, accuracy = model.evaluate(x_test_long, y_test_long)
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    y_pred = model.predict(x_test_long)
    y_pred_binary = (y_pred > 0.5).astype(int)

    report = classification_report(y_test_long, y_pred_binary)
    confusion_mat = confusion_matrix(y_test_long, y_pred_binary)

    results = {
        "Test accuracy": accuracy,
        "Test loss": loss,
        "Classification report": report,
        "Confusion matrix": confusion_mat.tolist(),  # convert numpy array to list for JSON serialization
        "Accuracy score": accuracy_score(y_test_long, y_pred_binary)
    }

    with open('reports/test_long_urls.txt', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print('Classification Report:')
    print(report)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test_long, y_pred_binary))

if __name__ == "__main__":
    test_model()
