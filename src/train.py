from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
from data_loader import load_data, preprocess_data, encode_labels

def train_model():
    # Load and preprocess data
    raw_train = load_data("data/train.txt")
    x_train, tokenizer = preprocess_data([line.split("\t")[1] for line in raw_train])
    y_train, encoder = encode_labels([line.split("\t")[0] for line in raw_train])

    # Build model
    voc_size = len(tokenizer.word_index.keys())
    model = build_model(voc_size, 2)  # Assuming 2 categories: 'phishing', 'legitimate'

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Setup model saving
    checkpoint_callback = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_loss')

    # Print summary of model
    print(model.summary())

    # Train model
    model.fit(x_train, y_train, batch_size=5000, epochs=30, callbacks=[checkpoint_callback])

if __name__ == "__main__":
    train_model()
