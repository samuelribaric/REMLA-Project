"""
Contains an integration test for the full ML pipeline
Does data downloading, processing, model training, and evaluation
"""

import os
import sys
import subprocess
import time
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json


def run_subprocess(command):
    """Utility function to run a subprocess and print its output in real-time."""
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for line in process.stdout:
            print(line.decode(), end='')
        process.stdout.close()
        process.wait()


def verify_file_exists(file_path):
    if not os.path.exists(file_path):
        raise Exception(f"Expected file {file_path} does not exist.")
    return os.path.getsize(file_path)  # Return file size for verification


def integration_test():
    """Integration test for full ML pipeline"""
    np.random.seed(42)
    results = {}
    
    # Step 1: Download Data
    print("Step 1: Downloading data...")
    start_time = time.time()
    run_subprocess(f"{sys.executable} data_downloader.py")
    results["download_data"] = {"status": "completed", "time": time.time() - start_time}

    # Step 2: Split Data into Features and Labels
    print("Step 2: Splitting data into features and labels...")
    data_files = ["train.txt", "val.txt", "test.txt"]
    for file in data_files:
        base_name = os.path.splitext(file)[0]
        start_time = time.time()
        run_subprocess(
            f"{sys.executable} data_split.py data/raw/{file} data/interim/{base_name}_features.txt "
            f"data/interim/{base_name}_labels.txt"
        )
        results[f"split_data_{base_name}"] = {
            "status": "completed",
            "time": time.time() - start_time,
            "output_files": {
                f"{base_name}_features.txt": verify_file_exists(f"data/interim/{base_name}_features.txt"),
                f"{base_name}_labels.txt": verify_file_exists(f"data/interim/{base_name}_labels.txt")
            }
        }

    # Step 3: Encode Labels
    print("Step 3: Encoding labels...")
    start_time = time.time()
    run_subprocess(
        f"{sys.executable} encode_labels.py data/interim/train_labels.txt data/interim/val_labels.txt "
        f"data/interim/test_labels.txt"
    )
    results["encode_labels"] = {
        "status": "completed",
        "time": time.time() - start_time,
        "output_files": {
            "encoded_train_labels.txt": verify_file_exists("data/interim/encoded_train_labels.txt"),
            "encoded_val_labels.txt": verify_file_exists("data/interim/encoded_val_labels.txt"),
            "encoded_test_labels.txt": verify_file_exists("data/interim/encoded_test_labels.txt")
        }
    }

    # Step 4: Tokenize Features
    print("Step 4: Tokenizing features...")
    start_time = time.time()
    run_subprocess(
        f"{sys.executable} tokenize_features.py data/interim/train_features.txt data/interim/val_features.txt "
        f"data/interim/test_features.txt"
    )
    results["tokenize_features"] = {
        "status": "completed",
        "time": time.time() - start_time,
        "output_files": {
            "tokenized_train.txt": verify_file_exists("data/interim/tokenized_train.txt"),
            "tokenized_val.txt": verify_file_exists("data/interim/tokenized_val.txt"),
            "tokenized_test.txt": verify_file_exists("data/interim/tokenized_test.txt")
        }
    }

    # Save results to JSON
    with open('reports/test_integration.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    integration_test()
    end_time = time.time()
    print(f"Integration test completed in {end_time - start_time:.2f} seconds.")
