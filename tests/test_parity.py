import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, brier_score_loss, roc_curve
import matplotlib.pyplot as plt


def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    demographics = np.loadtxt('data/interim/test_demographics.txt', dtype=str)
    return x_test, y_test, demographics


def plot_calibration_curve(y_true, y_prob, group):
    """Plots calibration curve for a demographic group"""
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{group}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration curve for {group}")
    plt.legend()
    plt.show()


def test_model_calibration():
    """Tests the model calibration for different demographic groups."""
    x_test, y_test, demographics = load_test_data()

    model = load_model("models/model.keras")

    unique_demographics = np.unique(demographics)

    for group in unique_demographics:
        group_indices = np.where(demographics == group)[0]
        x_test_group = x_test[group_indices]
        y_test_group = y_test[group_indices]

        y_prob = model.predict(x_test_group).flatten()
        brier_score = brier_score_loss(y_test_group, y_prob)

        print(f"\nBrier score for group {group}: {brier_score:.4f}")
        plot_calibration_curve(y_test_group, y_prob, group)


def test_equalized_odds():
    """Tests the model for equalized odds across different demographic groups."""
    x_test, y_test, demographics = load_test_data()

    model = load_model("models/model.keras")

    unique_demographics = np.unique(demographics)
    results = []

    for group in unique_demographics:
        group_indices = np.where(demographics == group)[0]
        x_test_group = x_test[group_indices]
        y_test_group = y_test[group_indices]

        y_prob = model.predict(x_test_group).flatten()
        y_pred_binary = (y_prob > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test_group, y_prob)
        results.append({'group': group, 'tpr': tpr, 'fpr': fpr})

        print(f"\nEqualized Odds for group {group}:")
        print(f"True Positive Rate: {tpr}")
        print(f"False Positive Rate: {fpr}")

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


def test_demographic_parity():
    """Tests the model for demographic parity across different demographic groups."""
    x_test, y_test, demographics = load_test_data()

    model = load_model("models/model.keras")

    unique_demographics = np.unique(demographics)

    for group in unique_demographics:
        group_indices = np.where(demographics == group)[0]
        x_test_group = x_test[group_indices]
        y_test_group = y_test[group_indices]

        y_prob = model.predict(x_test_group).flatten()
        y_pred_binary = (y_prob > 0.5).astype(int)

        prediction_rate = np.mean(y_pred_binary)

        print(f"\nDemographic Parity for group {group}:")
        print(f"Prediction Rate: {prediction_rate:.4f}")


if __name__ == "__main__":
    print("Testing Model Calibration...")
    test_model_calibration()
    print("\nTesting Equalized Odds...")
    test_equalized_odds()
    print("\nTesting Demographic Parity...")
    test_demographic_parity()
