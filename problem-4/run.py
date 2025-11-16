# run.py (Refactored for Clarity)

import os
import random
import numpy as np
import time
import sys

# import everything we need
from nsa import generate_detectors
from data_loader import load_sms_data
from encoder import encode_message
from evaluation import get_match_count, evaluate_with_thresholds, calculate_detector_statistics
from file_utils import save_detectors, save_results, load_detectors
import config

def split_data(
    ham_messages: list[str], 
    spam_messages: list[str], 
    self_ratio: float = 0.6, 
    validation_ratio: float = 0.2
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    print(f"Splitting data with self_ratio={self_ratio:.0%}, validation_ratio={validation_ratio:.0%}...")

    # Shuffle ham messages
    ham_messages_shuffled = ham_messages.copy()
    random.shuffle(ham_messages_shuffled)

    # Calculate split indices for ham
    self_end_idx = int(len(ham_messages_shuffled) * self_ratio)
    validation_end_idx = int(len(ham_messages_shuffled) * (self_ratio + validation_ratio))

    # Slice ham_messages into three parts
    self_set = ham_messages_shuffled[:self_end_idx]
    validation_ham = ham_messages_shuffled[self_end_idx:validation_end_idx]
    test_ham = ham_messages_shuffled[validation_end_idx:]

    # Shuffle spam messages
    spam_messages_shuffled = spam_messages.copy()
    random.shuffle(spam_messages_shuffled)

    # Calculate split index for spam (50/50 between validation and test)
    validation_end_idx_spam = int(len(spam_messages_shuffled) * 0.5)

    # Slice spam_messages into two parts
    validation_spam = spam_messages_shuffled[:validation_end_idx_spam]
    test_spam = spam_messages_shuffled[validation_end_idx_spam:]

    # Construct the final sets
    validation_set_labeled = [(message, 'ham') for message in validation_ham]
    validation_set_labeled.extend([(message, 'spam') for message in validation_spam])
    random.shuffle(validation_set_labeled)

    test_set_labeled = [(message, 'ham') for message in test_ham]
    test_set_labeled.extend([(message, 'spam') for message in test_spam])
    random.shuffle(test_set_labeled)

    print("Splitting complete.")
    print(f" -> 'Self' set size (for training): {len(self_set)} messages.")
    print(f" -> Validation set size: {len(validation_set_labeled)} messages.")
    print(f" -> Test set size (for evaluation): {len(test_set_labeled)} messages.")

    return self_set, validation_set_labeled, test_set_labeled

def get_or_generate_detectors(
    hyperparams: dict, 
    training_set: list[str], 
    seed: int
) -> tuple[list[np.ndarray], float, float | str]:
    # This new function handles all detector logic: loading, generating, saving.
    
    print("\n--- Checking for Pre-trained Detectors ---")
    detectors = load_detectors(hyperparams['R_CONTIGUOUS'], hyperparams['NUM_DETECTORS'], seed)
    
    training_time = 0.0
    acceptance_rate = "N/A (loaded from file)"
    
    if detectors is None:
        print("No detectors found, starting generation process...")
        
        # encode the self set
        encoded_self_set = [encode_message(msg, hyperparams['HASH_SIZE'], hyperparams['NGRAM_SIZE']) for msg in training_set]
        unique_self_tuples = {tuple(vec) for vec in encoded_self_set}
        unique_self_set = [np.array(t, dtype=np.uint8) for t in unique_self_tuples]
        
        # generate detectors
        start_time = time.time()
        detectors, acceptance_rate = generate_detectors(unique_self_set, hyperparams['NUM_DETECTORS'], hyperparams['HASH_SIZE'], hyperparams['R_CONTIGUOUS'])
        end_time = time.time()
        training_time = end_time - start_time
        
        # save the newly generated detectors
        save_detectors(detectors, hyperparams['R_CONTIGUOUS'], hyperparams['NUM_DETECTORS'], seed)
        
    return detectors, training_time, acceptance_rate

def run_single_experiment(hyperparams: dict):
    # This function is now much cleaner.
    
    # load and split data
    dataset_path = os.path.join('data', 'dataset', 'SMSSpamCollection')
    ham, spam = load_sms_data(dataset_path)
    if not ham or not spam: sys.exit("Could not load data.")
    self_training_set, validation_set, final_test_set = split_data(ham, spam)

    # get the detectors and training info
    detectors, training_time, acceptance_rate = get_or_generate_detectors(hyperparams, self_training_set, hyperparams['SEED'])

    # calculate detector validation statistics
    detector_stats = calculate_detector_statistics(
        detectors, 
        validation_set, 
        hyperparams['R_CONTIGUOUS'], 
        hyperparams['HASH_SIZE'], 
        hyperparams['NGRAM_SIZE']
    )

    # evaluate on the test set
    print(f"\n--- Starting Evaluation for R={hyperparams['R_CONTIGUOUS']} ---")
    test_results = []
    for message, actual_label in final_test_set:
        encoded_msg = encode_message(message, hyperparams['HASH_SIZE'], hyperparams['NGRAM_SIZE'])
        match_count = get_match_count(encoded_msg, detectors, hyperparams['R_CONTIGUOUS'])
        test_results.append((match_count, actual_label))
    
    # get results for all thresholds
    results_by_threshold = evaluate_with_thresholds(test_results, config.THRESHOLDS)

    # compile and save final results file
    final_results_package = {
        "hyperparameters": hyperparams,
        "training_info": {
            "training_time_seconds": round(training_time, 2),
            "acceptance_rate": acceptance_rate
        },
        "detector_validation_stats": detector_stats,
        "evaluation_by_threshold": results_by_threshold
    }
    save_results(final_results_package)

def main():
    # --- Main Experiment Loop ---
    print("===== STARTING EXPERIMENT SUITE =====")
    
    # Set seeds for reproducibility for both libraries
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    
    # iterate through all R values defined in the config
    for r_val in config.R_CONTIGUOUS_VALUES:
        print(f"\n{'='*10} RUNNING EXPERIMENT: R_CONTIGUOUS = {r_val} {'='*10}")
        
        # define the parameters for this specific run
        hyperparams_for_run = {
            "HASH_SIZE": config.HASH_SIZE,
            "NGRAM_SIZE": config.NGRAM_SIZE,
            "NUM_DETECTORS": config.NUM_DETECTORS,
            "R_CONTIGUOUS": r_val,
            "SEED": config.SEED
        }
        
        run_single_experiment(hyperparams_for_run)
        
    print(f"\n{'='*10} EXPERIMENT SUITE COMPLETE {'='*10}")

if __name__ == '__main__':
    main()