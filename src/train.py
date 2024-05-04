"""
This module trains the model using the training dataset.
"""

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from data_loader import load_and_preprocess_data
from model import build_model

def train_model():
    """Trains the model on the training dataset."""
    file_paths = {
        'train': 'data/train.txt',
        'val': 'data/val.txt'
    }
    data, char_index = load_and_preprocess_data(file_paths)
    x_train, y_train = data['train']
    x_val, y_val = data['val']

    model = build_model(len(char_index), 2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint_path = "models/best_model.keras"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    model.fit(x_train, y_train, batch_size=5000, epochs=30, validation_data=(x_val, y_val), 
              callbacks=[checkpoint_callback])

    print(model.summary())
    model.save("models/final_model.keras")

if __name__ == "__main__":
    train_model()
