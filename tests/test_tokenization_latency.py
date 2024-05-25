import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_test_data():
    with open('data/interim/test_features.txt', 'r', encoding="utf-8") as file:
        test_features = file.readlines()
    return test_features

def test_tokenization_latency():
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
        
        # Updated print statement to include tokenization rate
        print(f"\rTokenized {index}/{total_features} URLs", end='')

    print("\nTotal tokenization time: {:.2f} seconds, Average rate: {:.2f} URLs/sec".format(elapsed_time, rate))

    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    
    print(f"Min Time: {min_time:.5f} seconds")
    print(f"Max Time: {max_time:.5f} seconds")
    print(f"Average Time: {avg_time:.5f} seconds")
    print(f"Total evaluated: {total_features}")

if __name__ == "__main__":
    test_tokenization_latency()
