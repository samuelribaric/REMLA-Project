"""
This module provides functionality for loading and preprocessing data.
"""
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def load_and_preprocess_data(file_paths):
    """Loads data from a specified file path then Preprocesses text data by tokenizing 
    and padding sequences and finally Encodes labels using label encoder"""
    data = {}
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    raw_data = []

    for key, path in file_paths.items():
        with open(path, "r") as file:
            if key == 'train':
                lines = [line.strip() for line in file.readlines()[1:]]  
            else:
                lines = [line.strip() for line in file.readlines()] 

            data[key] = [line.split("\t") for line in lines]
            raw_data.extend([line.split("\t")[1] for line in lines])

    tokenizer.fit_on_texts(raw_data)
    sequence_length = 200

    for key in file_paths.keys():
        x = pad_sequences(tokenizer.texts_to_sequences([item[1] for item in data[key]]), maxlen=sequence_length)
        y = LabelEncoder().fit_transform([item[0] for item in data[key]])
        data[key] = (x, y)

    return data, tokenizer.word_index

