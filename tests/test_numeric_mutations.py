import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def load_test_data():
    """Loads test data from interim files"""
    x_test = np.loadtxt('data/interim/tokenized_test.txt', dtype=int)
    y_test = np.loadtxt('data/interim/encoded_test_labels.txt', dtype=int)
    return x_test, y_test

def load_tokenizer():
    """Loads the tokenizer from a pickle file"""
    with open("data/interim/tokenizer.pkl", 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

def mutate_url_sequence(sequence, tokenizer):
    """Mutates a sequence by replacing some tokens with similar-looking tokens."""
    # Mapping similar-looking characters
    char_to_token = tokenizer.word_index
    mutation_map = {char_to_token.get('a'): char_to_token.get('4'),
                    char_to_token.get('e'): char_to_token.get('3'),
                    char_to_token.get('i'): char_to_token.get('1'),
                    char_to_token.get('o'): char_to_token.get('0'),
                    char_to_token.get('l'): char_to_token.get('1')}
    return [mutation_map.get(token, token) for token in sequence]

def generate_mutant_data(x_test, tokenizer):
    """Generates mutant test data by mutating the original sequences."""
    x_test_mutant = np.array([mutate_url_sequence(sequence, tokenizer) for sequence in x_test])
    return x_test_mutant

def test_model():
    """Tests the saved model on original and mutant URLs in the test dataset."""
    x_test, y_test = load_test_data()
    
    # Print the total number of datapoints initially
    print("Total number of datapoints:", len(x_test))
    
    # Load the tokenizer
    tokenizer = load_tokenizer()

    # Generate mutant test data
    x_test_mutant = generate_mutant_data(x_test, tokenizer)

    model = load_model("models/model.keras")

    # Evaluate on original test data
    loss_original, accuracy_original = model.evaluate(x_test, y_test)
    print(f"Original Test accuracy: {accuracy_original}")
    print(f"Original Test loss: {loss_original}")

    y_pred_original = model.predict(x_test)
    y_pred_original_binary = (y_pred_original > 0.5).astype(int)

    report_original = classification_report(y_test, y_pred_original_binary)
    confusion_mat_original = confusion_matrix(y_test, y_pred_original_binary)

    print('Original Classification Report:')
    print(report_original)
    print('Original Confusion Matrix:')
    print(confusion_mat_original)
    print('Original Accuracy:', accuracy_score(y_test, y_pred_original_binary))

    # Evaluate on mutant test data
    loss_mutant, accuracy_mutant = model.evaluate(x_test_mutant, y_test)
    print(f"Mutant Test accuracy: {accuracy_mutant}")
    print(f"Mutant Test loss: {loss_mutant}")

    y_pred_mutant = model.predict(x_test_mutant)
    y_pred_mutant_binary = (y_pred_mutant > 0.5).astype(int)

    report_mutant = classification_report(y_test, y_pred_mutant_binary)
    confusion_mat_mutant = confusion_matrix(y_test, y_pred_mutant_binary)

    print('Mutant Classification Report:')
    print(report_mutant)
    print('Mutant Confusion Matrix:')
    print(confusion_mat_mutant)
    print('Mutant Accuracy:', accuracy_score(y_test, y_pred_mutant_binary))

    # Compare results
    print("Comparison of Results:")
    print(f"Accuracy Difference: {accuracy_original - accuracy_mutant}")
    print(f"Loss Difference: {loss_mutant - loss_original}")

if __name__ == "__main__":
    test_model()