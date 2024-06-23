"""Utility module for loading and splitting data into features and labels"""
import sys
from remla2024_team9_lib_ml import load_and_split_data


def main():
    """Main to be executed when module is run as a script"""
    input_path = sys.argv[1]
    features_output_path = sys.argv[2]
    labels_output_path = sys.argv[3]

    # Load data and skip header if needed
    with open(input_path, 'r', encoding="utf-8") as file:
        data = file.readlines()[1:]  # adjust index as necessary to skip headers

    # Split into features and labels
    features, labels = load_and_split_data(data)

    # Save features and labels to separate files
    with (open(features_output_path, 'w', encoding="utf-8") as f_output,
          open(labels_output_path, 'w', encoding="utf-8") as l_output):
        for feature in features:
            f_output.write(feature + "\n")
        for label in labels:
            l_output.write(label + "\n")


if __name__ == "__main__":
    main()
