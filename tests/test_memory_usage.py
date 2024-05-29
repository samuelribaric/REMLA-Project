import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import psutil
import time


def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test


def test_memory_usage():
    """Tests the memory usage of the model on the test dataset."""
    x_test, y_test = load_test_data()

    print("Total number of datapoints:", len(x_test))

    model = load_model("models/model.keras")

    process = psutil.Process()

    # Measure memory usage before evaluation
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

    start_time = time.time()
    loss, accuracy = model.evaluate(x_test, y_test)
    end_time = time.time()

    # Measure memory usage after evaluation
    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"Memory Usage Before Evaluation: {mem_before:.2f} MB")
    print(f"Memory Usage After Evaluation: {mem_after:.2f} MB")
    print(f"Memory Usage Increase: {mem_after - mem_before:.2f} MB")

    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    report = classification_report(y_test, y_pred_binary)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)

    print('Classification Report:')
    print(report)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))


if __name__ == "__main__":
    test_memory_usage()
