import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_labels(train_labels_path, val_labels_path, test_labels_path):
    # Load label data
    with open(train_labels_path, 'r') as file:
        train_labels = file.readlines()
    with open(val_labels_path, 'r') as file:
        val_labels = file.readlines()
    with open(test_labels_path, 'r') as file:
        test_labels = file.readlines()

    # Initialize and fit LabelEncoder
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform([label.strip() for label in train_labels])
    val_labels = encoder.transform([label.strip() for label in val_labels])
    test_labels = encoder.transform([label.strip() for label in test_labels])

    # Save encoded labels
    np.savetxt("data/interim/encoded_train_labels.txt", train_labels, fmt='%d')
    np.savetxt("data/interim/encoded_val_labels.txt", val_labels, fmt='%d')
    np.savetxt("data/interim/encoded_test_labels.txt", test_labels, fmt='%d')

def main():
    train_labels_path = sys.argv[1]
    val_labels_path = sys.argv[2]
    test_labels_path = sys.argv[3]
    encode_labels(train_labels_path, val_labels_path, test_labels_path)

if __name__ == "__main__":
    main()
