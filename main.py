from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load data
train = [line.strip() for line in open("/data/raw/train_sample.txt", "r").readlines()[1:]]
raw_x_train = [line.split("\t")[1] for line in train]
raw_y_train = [line.split("\t")[0] for line in train]

test = [line.strip() for line in open("/kaggle/input/dl-dataset/DL Dataset/test.txt", "r").readlines()]
raw_x_test = [line.split("\t")[1] for line in test]
raw_y_test = [line.split("\t")[0] for line in test]

val = [line.strip() for line in open("/kaggle/input/dl-dataset/DL Dataset/val.txt", "r").readlines()]
raw_x_val = [line.split("\t")[1] for line in val]
raw_y_val = [line.split("\t")[0] for line in val]

# Tokenize data
tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
char_index = tokenizer.word_index
sequence_length = 200
x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(raw_y_train)
y_val = encoder.transform(raw_y_val)
y_test = encoder.transform(raw_y_test)

# Model parameters
params = {
    'loss_function': 'binary_crossentropy',
    'optimizer': 'adam',
    'sequence_length': 200,
    'batch_train': 5000,
    'batch_test': 5000,
    'categories': ['phishing', 'legitimate'],
    'char_index': None,
    'epoch': 30,
    'embedding_dimension': 50,
    'dataset_dir': "../dataset/small_dataset/"
}

# Build model
model = Sequential()
voc_size = len(char_index.keys())
print("voc_size: {}".format(voc_size))
model.add(Embedding(voc_size + 1, 50))

model.add(Conv1D(128, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(128, 7, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, activation='tanh', padding='same'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, activation='tanh'))

# Compile model
model.compile(optimizer=params['optimizer'], loss=params['loss_function'])

# Print model summary
print(model.summary())

# Train the model
model.fit(x_train, y_train, batch_size=params['batch_train'], epochs=params['epoch'])