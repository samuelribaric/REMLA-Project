import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Daniel: We are using batch processing, because else I run into memory issues.

def encode_labels(input_path, output_path, batch_size=10000):
    with open(input_path, 'r') as file:
        labels = []
        encoder = LabelEncoder()
        # Open the output file once and append to it
        with open(output_path, 'ab') as output_file:  # Using 'ab' to append in binary mode
            for line in file:
                labels.append(line.strip())
                if len(labels) >= batch_size:
                    # Process the current batch
                    encoded_labels = encoder.fit_transform(labels)
                    np.savetxt(output_file, encoded_labels, fmt='%d')
                    labels = []
            # Process any remaining labels
            if labels:
                encoded_labels = encoder.fit_transform(labels)
                np.savetxt(output_file, encoded_labels, fmt='%d')


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    encode_labels(input_path, output_path)

if __name__ == "__main__":
    main()
