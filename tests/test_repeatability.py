"""Utility module for testing model on the test dataset"""
import numpy as np
from keras.models import load_model
import json

def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test

def test_model_repeatability(runs=5, sample_size=0.5):
    """Tests the model multiple times with randomly selected subsets of test data."""
    np.random.seed(42)  # Set seed for reproducibility
    accuracies = []
    losses = []

    x_test, y_test = load_test_data()
    model = load_model("models/model.keras")
    total_samples = int(len(x_test) * sample_size)
    for _ in range(runs):
        # Randomly sample the test data with replacement
        indices = np.random.choice(len(x_test), total_samples, replace=True)
        x_test_sample = x_test[indices]
        y_test_sample = y_test[indices]

        loss, accuracy = model.evaluate(x_test_sample, y_test_sample)
        accuracies.append(accuracy)
        losses.append(loss)

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)

    variability_warning = bool(std_accuracy > 0.01 or std_loss > 0.01)

    results = {
        "Average Test Accuracy": float(avg_accuracy),
        "Standard Deviation of Accuracy": float(std_accuracy),
        "Average Test Loss": float(avg_loss),
        "Standard Deviation of Loss": float(std_loss),
        "Variability Warning": variability_warning
    }

    with open('reports/test_repeatability.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Average Test Accuracy: {avg_accuracy}")
    print(f"Standard Deviation of Accuracy: {std_accuracy}")
    print(f"Average Test Loss: {avg_loss}")
    print(f"Standard Deviation of Loss: {std_loss}")

    if variability_warning:
        print("Warning: Model results show significant variability.")
    else:
        print("Model results are consistent.")

if __name__ == "__main__":
    test_model_repeatability()
