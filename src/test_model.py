import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def load_test_data():
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test

def test_model():
    """Tests the saved model on the test dataset."""
    x_test, y_test = load_test_data()

    model = load_model("models/model.keras")

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Save reports to files
    with open('reports/test_accuracy.txt', 'w') as f:
        f.write(f"Test accuracy: {accuracy}\n")

    with open('reports/test_loss.txt', 'w') as f:
        f.write(f"Test loss: {loss}\n")

    report = classification_report(y_test, y_pred_binary)
    with open('reports/classification_report.txt', 'w') as f:
        f.write('Classification Report:\n')
        f.write(report)

    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    with open('reports/confusion_matrix.txt', 'w') as f:
        f.write('Confusion Matrix:\n')
        f.write(str(confusion_mat))

    print('Classification Report:')
    print(report)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))

if __name__ == "__main__":
    test_model()
