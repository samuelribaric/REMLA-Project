"""
This module tests the trained model on test data.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_and_preprocess_data
from model import build_model

def test_model():
    """Tests the saved model on the test dataset."""
    file_paths = {'test': 'data/test.txt'}
    data, char_index = load_and_preprocess_data(file_paths)
    x_test, y_test = data['test']

    model = build_model(len(char_index), 2, input_length=x_test.shape[1])
    model.load_weights("models/best_model.keras")

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))

if __name__ == "__main__":
    test_model()
