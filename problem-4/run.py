# run.py (Optimized)

import os
import random
import numpy as np
import time
import sys

from nsa import generate_detectors
from data_loader import load_sms_data
from encoder import encode_message
from evaluation import classify_message, evaluate_performance

def split_data(
    ham_messages: list[str], 
    spam_messages: list[str], 
    train_ratio: float = 0.7, 
    seed: int = 42
) -> tuple[list[str], list[tuple[str, str]]]:
    print(f"Splitting data with a {train_ratio:.0%} train ratio and random seed {seed}...")

    random.seed(seed)
    # create a copy of the ham messages to shuffle
    ham_messages_shuffled = ham_messages.copy()
    # shuffle the copy
    random.shuffle(ham_messages_shuffled)
    # split the copy into a training set and a test set
    # the training set is the first 70% of the shuffled messages
    # the test set is the last 30% of the shuffled messages
    split_index = int(len(ham_messages_shuffled) * train_ratio)
    self_set = ham_messages_shuffled[:split_index]
    test_ham = ham_messages_shuffled[split_index:]
    # create a test set with the spam messages
    test_set_labeled = [(message, 'ham') for message in test_ham]
    test_set_labeled.extend([(message, 'spam') for message in spam_messages])
    # shuffle the test set
    random.shuffle(test_set_labeled)
    print("Splitting complete.")
    print(f" -> 'Self' set size (for training): {len(self_set)} messages.")
    print(f" -> Test set size (for evaluation): {len(test_set_labeled)} messages.")
    return self_set, test_set_labeled


if __name__ == '__main__':
    # optimized hyperparameters, increased r_contiguous significantly because 
    # it was SOOO slow
    HASH_SIZE = 256
    NGRAM_SIZE = 3
    NUM_DETECTORS = 2000
    R_CONTIGUOUS = 20 

    # 1. load the data
    dataset_path = os.path.join('data', 'dataset', 'SMSSpamCollection')
    ham, spam = load_sms_data(dataset_path)

    if not ham or not spam:
        print("\nCould not proceed without data. Exiting.")
        exit(1)

    # 2. split the data
    self_training_set, final_test_set = split_data(ham, spam)

    # 3. encode the 'self' set for training
    print(f"\nEncoding 'self' set with HASH_SIZE={HASH_SIZE} and NGRAM_SIZE={NGRAM_SIZE}...")
    encoded_self_set = [
        encode_message(msg, HASH_SIZE, NGRAM_SIZE)
        for msg in self_training_set
    ]
    print(f"Encoding complete. {len(encoded_self_set)} messages encoded.")
    
    # de-duplicate the encoded self set to speed up the algorithm
    print("\nOptimizing self set by removing duplicates...")
    # We convert each numpy array to a tuple to make it hashable, then add to a set.
    unique_self_tuples = {tuple(vec) for vec in encoded_self_set}
    # Convert back to a list of numpy arrays for the algorithm.
    unique_self_set = [np.array(t, dtype=np.uint8) for t in unique_self_tuples]
    print(f"Optimization complete. Unique self vectors: {len(unique_self_set)} (reduced from {len(encoded_self_set)}).")

    # 4. generate detectors using the OPTIMIZED self set
    print(f"\n--- Starting NSA Training ---")
    start_time = time.time()

    detectors = generate_detectors(
        unique_self_set,
        NUM_DETECTORS,
        HASH_SIZE,
        R_CONTIGUOUS
    )

    end_time = time.time()
    
    print(f"\n--- Training complete in {end_time - start_time:.2f} seconds ---")

    # sanity check
    if not detectors:
        print("No detectors were generated. This is unexpected.")
        exit(1)

    print(f"Created {len(detectors)} detectors.")
    # 5. evaluate the detectors
    print(f"Starting eval on {len(final_test_set)} test messages...")
    predictions = []
    
    for message, actual_label in final_test_set:
        # encode the message
        encoded_message = encode_message(message, HASH_SIZE, NGRAM_SIZE)
        # get the prediction
        predicted_label = classify_message(encoded_message, detectors, R_CONTIGUOUS)
        predictions.append((actual_label, predicted_label))

    evaluate_performance(predictions)

    
    print("\nProgram Finished.")