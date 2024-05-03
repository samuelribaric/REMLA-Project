import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

def load_data(filepath):
    return np.loadtxt(filepath, dtype=int)

def train_model(params, x_train_path, y_train_path):
    # Load training data
    x_train = load_data(x_train_path)
    y_train = load_data(y_train_path)
    
    # Model architecture
    model = Sequential()
    model.add(Embedding(params['voc_size'] + 1, params['embedding_dimension'])) # CHECK

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

    # Train model
    model.fit(x_train, y_train, batch_size=params['batch_train'], epochs=params['epoch'])

    # Save model
    model.save('models/model.h5')

def main():
    params = {
        'loss_function': 'binary_crossentropy',
        'optimizer': 'adam',
        'embedding_dimension': 50,
        'batch_train': 5000,
        'epoch': 30,
        'voc_size': len(open('data/interim/tokenizer.json').read())  # Update this to properly count vocabulary
    }
    x_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    train_model(params, x_train_path, y_train_path)

if __name__ == "__main__":
    main()
