import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Daniel: We are using batch processing, because else I run into memory issues.

def encode_labels(input_path, output_path):
    # Load labels from the file
    with open(input_path, 'r') as file:
        labels = [line.strip() for line in file]

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Save the encoded labels
    np.savetxt(output_path, encoded_labels, fmt='%d')


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    encode_labels(input_path, output_path)

if __name__ == "__main__":
    main()
