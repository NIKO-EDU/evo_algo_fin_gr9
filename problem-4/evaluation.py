# evaluation.py

import numpy as np
from nsa import match_r_contiguous # we need the same matching rule from training

def classify_message(
    encoded_message: np.ndarray,
    detectors: list[np.ndarray],
    r_value: int
) -> str:
    # Classifies a single encoded message as 'ham' or 'spam'.
    # A message is spam if it matches ANY detector.

    for detector in detectors:
        if match_r_contiguous(encoded_message, detector, r_value):
            return 'spam' # Found a match, classify as non-self (spam) and stop early.
    
    return 'ham' # No detectors matched, classify as self (ham).

def evaluate_performance(predictions: list[tuple[str, str]]):
    # Calculates and prints the performance metrics from a list of (actual, predicted) labels.

    # initialize counters
    true_positives = 0 # spam predicted as spam
    false_positives = 0 # ham predicted as spam
    true_negatives = 0 # ham predicted as ham
    false_negatives = 0 # spam predicted as ham

    for actual_label, predicted_label in predictions:
        if actual_label == 'spam' and predicted_label == 'spam':
            true_positives += 1
        elif actual_label == 'ham' and predicted_label == 'spam':
            false_positives += 1
        elif actual_label == 'ham' and predicted_label == 'ham':
            true_negatives += 1
        elif actual_label == 'spam' and predicted_label == 'ham':
            false_negatives += 1

    # calculate metrics, handling division by zero for safety
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # as per requirements, calculate false positive rate on ham
    fp_rate_on_ham = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Confusion Matrix:")
    print(f"  - True Positives (Spam as Spam): {true_positives}")
    print(f"  - False Positives (Ham as Spam): {false_positives}")
    print(f"  - True Negatives (Ham as Ham):   {true_negatives}")
    print(f"  - False Negatives (Spam as Ham): {false_negatives}")
    
    print("\nPerformance Metrics for SPAM class:")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1_score:.4f} (Primary Metric)")
    
    print("\nPerformance Metrics for HAM class:")
    print(f"  - False Positive Rate on Ham: {fp_rate_on_ham:.4f}")