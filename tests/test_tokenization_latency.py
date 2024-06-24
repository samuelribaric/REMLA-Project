"""
This module tests the latency of tokenizing URLs using a Keras Tokenizer.
"""

import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def load_test_data():
    """Loads test features from an interim file"""
    with open('data/interim/test_features.txt', 'r', encoding="utf-8") as file:
        test_features = file.readlines()
    return test_features


def test_tokenization_latency():
    """Tests the tokenization latency for URLs in the test dataset."""
    test_features = load_test_data()
    sequence_length = 200

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(test_features)

    times = []
    total_features = len(test_features)
    print(f"Starting tokenization of {total_features} URLs.")
    start_total_time = time.time()  # Start total time before the loop

    for index, feature in enumerate(test_features, 1):
        start_time = time.time()
        _ = pad_sequences(tokenizer.texts_to_sequences([feature]), maxlen=sequence_length)
        end_time = time.time()

        times.append(end_time - start_time)
        elapsed_time = max(time.time() - start_total_time, 0.001)
        rate = index / elapsed_time  # Calculate URLs tokenized per second

        print(f"\rTokenized {index}/{total_features} URLs", end='')

    print(f"\nTotal tokenization time: {elapsed_time:.2f} seconds, Average rate: {rate:.2f} URLs/sec")

    results = {
        "Total tokenization time": elapsed_time,
        "Average rate": rate,
        "Min Time": min(times),
        "Max Time": max(times),
        "Average Time": sum(times) / len(times),
        "Total evaluated": total_features
    }

    with open('reports/test_tokenization_latency.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Min Time: {min(times):.5f} seconds")
    print(f"Max Time: {max(times):.5f} seconds")
    print(f"Average Time: {sum(times) / len(times):.5f} seconds")
    print(f"Total evaluated: {total_features}")


if __name__ == "__main__":
    test_tokenization_latency()
