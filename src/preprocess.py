import sys
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(filepath, sequences_output_path, tokenizer_output_path, sequence_length=200):
    # Load data from the file
    with open(filepath, 'r') as file:
        data = [line.strip() for line in file]

    # Initialize tokenizer or load existing
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(data)
    
    # Convert texts to sequences and pad them
    sequences = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=sequence_length)

    # Save the preprocessed sequences and tokenizer for later use
    with open(sequences_output_path, 'w') as file:
        for item in sequences:
            file.write(' '.join(map(str, item)) + '\n')
    
    # Save the tokenizer as JSON
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_output_path, 'w') as file:
        file.write(json.dumps(tokenizer_json))

def main():
    input_path = sys.argv[1]
    sequences_output_path = sys.argv[2]
    tokenizer_output_path = sys.argv[3]
    preprocess_data(input_path, sequences_output_path, tokenizer_output_path)

if __name__ == "__main__":
    main()
