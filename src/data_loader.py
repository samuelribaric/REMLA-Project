from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    with open(filepath, "r") as file:
        data = [line.strip() for line in file.readlines()]
    return data

def preprocess_data(data, tokenizer=None, sequence_length=200):
    if tokenizer is None:
        tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        tokenizer.fit_on_texts(data)
    sequences = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=sequence_length)
    return sequences, tokenizer

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder
