"""Utility module for loading and preprocessing text data"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_paths):
    """function reads data from `file_paths` dictionary, preprocesses it, and returns dictionary containing
    preprocessed data and word index from the Tokenizer."""
    processed_data = {}
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    raw_data = []

    for key, path in file_paths.items():
        with open(path, "r", encoding="utf-8") as file:
            if key == 'train':
                lines = [line.strip() for line in file.readlines()[1:]]
            else:
                lines = [line.strip() for line in file.readlines()]

            processed_data[key] = [line.split("\t") for line in lines]
            raw_data.extend([line.split("\t")[1] for line in lines])

    tokenizer.fit_on_texts(raw_data)
    sequence_length = 200

    for key in file_paths.keys():
        x_axis = pad_sequences(tokenizer.texts_to_sequences([item[1] for item in data[key]]), maxlen=sequence_length)
        y_axis = LabelEncoder().fit_transform([item[0] for item in data[key]])
        processed_data[key] = (x_axis, y_axis)

    return processed_data, tokenizer.word_index


if __name__ == "__main__":
    paths = {
        'train': 'data/train.txt',
        'test': 'data/test.txt',
        'val': 'data/val.txt'
    }
    data, char_index = load_and_preprocess_data(paths)

