# generate_coverage_data.py

import os
import json
import random
import numpy as np
import time

# Import necessary components from your project
from config import SEED
from data_loader import load_sms_data
from encoder import encode_message
from file_utils import load_detectors
from run import split_data
from evaluation import get_match_count

# --- Configuration ---
R_VALUE = 14
NUM_DETECTORS = 2000
# We will fix the classification threshold to 1 for this analysis.
# A match with *any* detector flags the message as spam.
DETECTION_THRESHOLD = 1
# Define the steps for the x-axis of our graph
DETECTOR_STEPS = list(range(100, NUM_DETECTORS + 1, 100))

def generate_data():
    """
    Evaluates the test set against growing subsets of detectors to generate
    data for the recall vs. number of detectors curve.
    """
    print("===== GENERATING DETECTOR COVERAGE DATA =====")
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. Load the full set of detectors
    print("Loading pre-trained detectors...")
    detectors = load_detectors(R_VALUE, NUM_DETECTORS, SEED)
    if detectors is None:
        print("Detectors not found. Please run the main experiment first.")
        return

    # 2. Re-create the exact same test set
    print("Loading and splitting data to get the test set...")
    ham, spam = load_sms_data(os.path.join('data', 'dataset', 'SMSSpamCollection'))
    _, _, test_set = split_data(ham, spam)

    # 3. Loop through detector subsets and calculate recall
    coverage_results = []
    total_spam_in_test_set = sum(1 for _, label in test_set if label == 'spam')
    
    start_time = time.time()
    for step in DETECTOR_STEPS:
        detector_subset = detectors[:step]
        true_positives = 0
        
        print(f"  -> Testing with {step}/{NUM_DETECTORS} detectors...")

        for message, actual_label in test_set:
            if actual_label == 'spam':
                encoded_msg = encode_message(message, 512, 3) # Using the known hyperparams
                match_count = get_match_count(encoded_msg, detector_subset, R_VALUE)
                
                if match_count >= DETECTION_THRESHOLD:
                    true_positives += 1
        
        recall = true_positives / total_spam_in_test_set if total_spam_in_test_set > 0 else 0
        coverage_results.append({'num_detectors': step, 'recall': recall})

    end_time = time.time()
    print(f"Data generation complete in {end_time - start_time:.2f} seconds.")

    # 4. Save the results to a new file
    os.makedirs('data/results', exist_ok=True)
    save_path = os.path.join('data/results', 'coverage_data.json')
    with open(save_path, 'w') as f:
        json.dump(coverage_results, f, indent=4)
        
    print(f"Coverage data successfully saved to '{save_path}'")

if __name__ == '__main__':
    generate_data()