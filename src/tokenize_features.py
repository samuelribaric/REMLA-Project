import sys
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_features(train_path, val_path, test_path, sequence_length=200):
    # Load feature data
    with open(train_path, 'r') as file:
        train_features = file.readlines()
    with open(val_path, 'r') as file:
        val_features = file.readlines()
    with open(test_path, 'r') as file:
        test_features = file.readlines()

    # Initialize tokenizer
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(train_features + val_features + test_features)

    # Tokenize and pad sequences
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_features), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_features), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_features), maxlen=sequence_length)

    # Save tokenized data
    np.savetxt("data/interim/tokenized_train.txt", x_train, fmt='%d')
    np.savetxt("data/interim/tokenized_val.txt", x_val, fmt='%d')
    np.savetxt("data/interim/tokenized_test.txt", x_test, fmt='%d')

    # Save the tokenizer as JSON
    tokenizer_json = tokenizer.to_json()
    with open("data/interim/tokenizer.json", 'w') as file:
        file.write(json.dumps(tokenizer_json))

def main():
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    tokenize_features(train_path, val_path, test_path)

if __name__ == "__main__":
    main()
