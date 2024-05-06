"""import module docstring here"""
import sys
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_features(train_path, val_path, test_path, sequence_length=200):
    """import method docstring here"""
    # Load feature data
    with open(train_path, 'r', encoding="utf-8") as file:
        train_features = file.readlines()
    with open(val_path, 'r', encoding="utf-8") as file:
        val_features = file.readlines()
    with open(test_path, 'r', encoding="utf-8") as file:
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

    # Save the tokenizer
    with open("data/interim/tokenizer.pkl", 'wb', encoding="utf-8") as file:
        pickle.dump(tokenizer, file)

def main():
    """import method docstring here"""
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    tokenize_features(train_path, val_path, test_path)

if __name__ == "__main__":
    main()
