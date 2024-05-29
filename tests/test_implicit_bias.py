import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    demographics = np.loadtxt('data/interim/test_demographics.txt', dtype=str)
    return x_test, y_test, demographics


def test_implicit_bias():
    """Tests the model for implicit bias across different demographic groups."""
    x_test, y_test, demographics = load_test_data()

    model = load_model("models/model.keras")

    unique_demographics = np.unique(demographics)
    results = []

    for group in unique_demographics:
        group_indices = np.where(demographics == group)[0]
        x_test_group = x_test[group_indices]
        y_test_group = y_test[group_indices]

        loss, accuracy = model.evaluate(x_test_group, y_test_group, verbose=0)
        y_pred = model.predict(x_test_group)
        y_pred_binary = (y_pred > 0.5).astype(int)

        report = classification_report(y_test_group, y_pred_binary, output_dict=True)
        confusion_mat = confusion_matrix(y_test_group, y_pred_binary)
        accuracy = accuracy_score(y_test_group, y_pred_binary)

        results.append({
            'group': group,
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_mat
        })

        print(f"\nResults for group: {group}")
        print(f"Test accuracy: {accuracy}")
        print(f"Test loss: {loss}")
        print('Classification Report:')
        print(classification_report(y_test_group, y_pred_binary))
        print('Confusion Matrix:')
        print(confusion_mat)

    return results


if __name__ == "__main__":
    results = test_implicit_bias()
    # Save the results to a file if needed
    import json

    with open('implicit_bias_results.json', 'w') as f:
        json.dump(results, f, indent=4)
