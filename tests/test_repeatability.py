"""Utility module for testing model on the test dataset"""
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test

def test_model_repeatability(runs=5, sample_size=0.5):
    """Tests the model multiple times with randomly selected subsets of test data."""
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
    
    print(f"Average Test Accuracy: {np.mean(accuracies)}")
    print(f"Standard Deviation of Accuracy: {np.std(accuracies)}")
    print(f"Average Test Loss: {np.mean(losses)}")
    print(f"Standard Deviation of Loss: {np.std(losses)}")

    if np.std(accuracies) > 0.01 or np.std(losses) > 0.01:
        print("Warning: Model results show significant variability.")
    else:
        print("Model results are consistent.")

if __name__ == "__main__":
    test_model_repeatability()
