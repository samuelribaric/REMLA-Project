"""
This module tests the equalized odds across different demographic groups.
"""

import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import json

def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    demographics = np.loadtxt('data/interim/test_demographics.txt', dtype=str)
    return x_test, y_test, demographics

def test_equalized_odds():
    """Tests the model for equalized odds across different demographic groups."""
    np.random.seed(42)  # Set seed for reproducibility
    x_test, y_test, demographics = load_test_data()

    model = load_model("models/model.keras")

    unique_demographics = np.unique(demographics)
    results = []

    for group in unique_demographics:
        group_indices = np.where(demographics == group)[0]
        x_test_group = x_test[group_indices]
        y_test_group = y_test[group_indices]

        y_prob = model.predict(x_test_group).flatten()
        fpr, tpr, _ = roc_curve(y_test_group, y_prob)
        results.append({'group': group, 'tpr': tpr.tolist(), 'fpr': fpr.tolist()})

        print(f"\nEqualized Odds for group {group}:")
        print(f"True Positive Rate: {tpr}")
        print(f"False Positive Rate: {fpr}")

    # Save results to JSON
    with open('reports/test_parity.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    # Plotting TPR and FPR for each group
    plt.figure()
    for result in results:
        plt.plot(result['fpr'], result['tpr'], label=f"Group {result['group']}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Equalized Odds')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_equalized_odds()
