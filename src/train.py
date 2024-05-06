"""Utility module for training the text classification model"""
import os
from keras.callbacks import ModelCheckpoint
from data_loader import load_and_preprocess_data
from model import build_model

def train_model():
    """Trains the classification model using training and validation datasets"""
    file_paths = {
        'train': 'data/train.txt',
        'val': 'data/val.txt'
    }
    data, char_index = load_and_preprocess_data(file_paths)
    x_train, y_train = data['train']
    x_val, y_val = data['val']

    model = build_model(len(char_index), 2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint_path = "models/best_model.train"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', save_format='h5')
    model.fit(x_train, y_train, batch_size=5000, epochs=30, validation_data=(x_val, y_val),
              callbacks=[checkpoint_callback])

    print(model.summary())

    # Save the entire model after training
    model.save("models/final_model.keras")

if __name__ == "__main__":
    train_model()
