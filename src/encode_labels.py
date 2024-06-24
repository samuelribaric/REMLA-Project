"""Utility module for encoding labels"""
import sys
import numpy as np

from remla2024_team9_lib_ml import encode_labels


def main():
    """Main to be executed when module is run as a script"""
    train_labels_path = sys.argv[1]
    val_labels_path = sys.argv[2]
    test_labels_path = sys.argv[3]

    with open(train_labels_path, 'r', encoding="utf-8") as file:
        train_labels = file.readlines()
    with open(val_labels_path, 'r', encoding="utf-8") as file:
        val_labels = file.readlines()
    with open(test_labels_path, 'r', encoding="utf-8") as file:
        test_labels = file.readlines()

    train_labels, val_labels, test_labels = encode_labels(train_labels, val_labels, test_labels)
    # Save encoded labels
    np.savetxt("data/interim/encoded_train_labels.txt", train_labels, fmt='%d')
    np.savetxt("data/interim/encoded_val_labels.txt", val_labels, fmt='%d')
    np.savetxt("data/interim/encoded_test_labels.txt", test_labels, fmt='%d')

if __name__ == "__main__":
    main()
