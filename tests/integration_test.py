import os
import sys
import subprocess
import time
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def run_subprocess(command):
    """Utility function to run a subprocess and print its output in real-time."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode(), end='')
    process.stdout.close()
    process.wait()


def integration_test():
    """Integration test for full ML pipeline"""
    #Download Data
    print("Step 1: Downloading data...")
    run_subprocess(f"{sys.executable} data_downloader.py")

    # Step 2: Split Data into Features and Labels
    print("Step 2: Splitting data into features and labels...")
    data_files = ["train.txt", "val.txt", "test.txt"]
    for file in data_files:
        base_name = os.path.splitext(file)[0]
        run_subprocess(
            f"{sys.executable} data_split.py data/raw/{file} data/interim/{base_name}_features.txt data/interim/{base_name}_labels.txt")

    # Step 3: Encode Labels
    print("Step 3: Encoding labels...")
    run_subprocess(
        f"{sys.executable} encode_labels.py data/interim/train_labels.txt data/interim/val_labels.txt data/interim/test_labels.txt")

    # Step 4: Tokenize Features
    print("Step 4: Tokenizing features...")
    run_subprocess(
        f"{sys.executable} tokenize_features.py data/interim/train_features.txt data/interim/val_features.txt data/interim/test_features.txt")

    # Step 5: Train Model
    print("Step 5: Training the model...")
    run_subprocess(f"{sys.executable} train_model.py")

    # Step 6: Evaluate Model
    print("Step 6: Evaluating the model...")
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    model = load_model("models/model.keras")

    loss, accuracy = model.evaluate(x_test, y_test)
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
    start_time = time.time()
    integration_test()
    end_time = time.time()
    print(f"Integration test completed in {end_time - start_time:.2f} seconds.")
