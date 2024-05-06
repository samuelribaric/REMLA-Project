"""Utility module for testing a trained model"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from data_loader import load_and_preprocess_data

def test_model():
    """Loads a trained model and evaluates its performance on the test dataset."""
    file_path = {
        'test': 'data/test.txt'
    }
    data, _ = load_and_preprocess_data(file_path)
    x_test, y_test = data['test']

    # Load the entire model from the .keras file
    model = load_model("models/best_model.keras")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")
    print(f"Test loss: {loss}")

    y_pred = model.predict(x_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    # Generate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))


if __name__ == "__main__":
    test_model()

