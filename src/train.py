import sys
import json
import numpy as np
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def train_model(tokenized_data_path, encoded_labels_path, tokenizer_path, model_path):
    # Load data
    x_train = np.loadtxt(tokenized_data_path, dtype=int)
    y_train = np.loadtxt(encoded_labels_path, dtype=int)
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Build model
    voc_size = len(tokenizer.word_index.keys())
    model = build_model(voc_size, 2)  # Assuming 2 categories: 'phishing', 'legitimate'

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Setup model saving
    checkpoint_callback = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    # Print summary of model
    print(model.summary())

    # Train model
    model.fit(x_train, y_train, batch_size=5000, epochs=30, callbacks=[checkpoint_callback])

if __name__ == "__main__":
    import sys
    tokenized_data_path = sys.argv[1]
    encoded_labels_path = sys.argv[2]
    tokenizer_path = sys.argv[3]
    model_path = sys.argv[4]
    train_model(tokenized_data_path, encoded_labels_path, tokenizer_path, model_path)
