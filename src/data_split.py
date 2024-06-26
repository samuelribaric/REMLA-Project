"""Utility module for loading and splitting data into features and labels"""
import sys

def load_and_split_data(input_file, features_output_file, labels_output_file):
    """Loads data from `input_file`, splits into features and labels, saves them to separate output files"""
    # Load data and skip header if needed
    with open(input_file, 'r', encoding="utf-8") as file:
        data = file.readlines()[1:]  # adjust index as necessary to skip headers

    # Split into features and labels
    features = [line.split("\t")[1].strip() for line in data]
    labels = [line.split("\t")[0].strip() for line in data]

    # Save features and labels to separate files
    with (open(features_output_file, 'w', encoding="utf-8") as f_output,
          open(labels_output_file, 'w', encoding="utf-8") as l_output):
        for feature in features:
            f_output.write(feature + "\n")
        for label in labels:
            l_output.write(label + "\n")

def main():
    """Main to be executed when module is run as a script"""
    input_path = sys.argv[1]
    features_output_path = sys.argv[2]
    labels_output_path = sys.argv[3]
    load_and_split_data(input_path, features_output_path, labels_output_path)

if __name__ == "__main__":
    main()
