import sys

def load_data(filepath):
    """ Load data from a file into a list of lines. """
    with open(filepath, "r") as file:
        data = [line.strip() for line in file.readlines()]
    return data

def save_data(data, filepath):
    """ Save processed data back to a file. """
    with open(filepath, "w") as file:
        for item in data:
            file.write(item + "\n")

def main():
    """ Entry point for the command-line interface. """
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    data = load_data(input_path)
    save_data(data, output_path)

if __name__ == "__main__":
    main()
