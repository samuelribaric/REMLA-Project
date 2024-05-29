"""import module docstring here"""
import sys
import pickle
import numpy as np

from remla2024_team9_lib_ml import tokenize_features


def main():
    """import method docstring here"""
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]


    with open(train_path, 'r', encoding="utf-8") as file:
        train_features = file.readlines()
    with open(val_path, 'r', encoding="utf-8") as file:
        val_features = file.readlines()
    with open(test_path, 'r', encoding="utf-8") as file:
        test_features = file.readlines()

    x_train, x_val, x_test, tokenizer = tokenize_features(train_features, val_features, test_features)

    # Save tokenized data
    np.savetxt("data/interim/tokenized_train.txt", x_train, fmt='%d')
    np.savetxt("data/interim/tokenized_val.txt", x_val, fmt='%d')
    np.savetxt("data/interim/tokenized_test.txt", x_test, fmt='%d')

    # Save the tokenizer
    with open("data/interim/tokenizer.pkl", 'wb') as file:
        pickle.dump(tokenizer, file)


if __name__ == "__main__":
    main()
