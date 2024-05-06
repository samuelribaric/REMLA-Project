import json
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
import keras

def load_data():
    # Load tokenized features
    x_train = np.loadtxt('data/interim/tokenized_train.txt', dtype=int)
    x_val = np.loadtxt('data/interim/tokenized_val.txt', dtype=int)

    # Load encoded labels
    y_train = np.loadtxt('data/interim/encoded_train_labels.txt', dtype=int)
    y_val = np.loadtxt('data/interim/encoded_val_labels.txt', dtype=int)

    # Load tokenizer
    with open('data/interim/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    return x_train, y_train, x_val, y_val, tokenizer

def main(params):
    print("Loading data...")
    x_train, y_train, x_val, y_val, tokenizer = load_data()

    print("Data loaded. Building model...")
    char_index = tokenizer.word_index
    voc_size =  len(char_index.keys())
    input_length = x_train.shape[1]
    model = create_model(voc_size, len(params['categories']), input_length)

    print("Model built. Compiling model...")
    model.compile(
        loss=params['loss_function'],
        optimizer=params['optimizer'],
        metrics=['accuracy']
    )

    print("Model compiled. Starting training...")
    hist = model.fit(
        x_train, y_train,
        batch_size=params['batch_train'],
        epochs=params['epoch'],
        shuffle=True,
        validation_data=(x_val, y_val)
    )

    # Save model
    print("Training complete. Saving model...")
    model.save('models/model.keras')

    # Save training metrics
    print("Saving training metrics...")
    metrics = {
        'loss': hist.history['loss'],
        'accuracy': hist.history['accuracy'],
        'val_loss': hist.history['val_loss'],
        'val_accuracy': hist.history['val_accuracy']
    }

    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Plot training metrics
    print("Plotting training metrics...")
    plt.figure()
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('reports/loss_plot.svg')
    plt.close()

if __name__ == "__main__":
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)
    main(params)
