# evaluation.py

import numpy as np
from typing import Any
from nsa import match_r_contiguous # we need the same matching rule from training
from encoder import encode_message

def get_match_count(
    encoded_message: np.ndarray,
    detectors: list[np.ndarray],
    r_value: int
) -> int:
    match_count = 0
    for detector in detectors:
        if match_r_contiguous(encoded_message, detector, r_value):
            match_count += 1
    return match_count

def calculate_detector_statistics(
    detectors: list[np.ndarray],
    validation_set: list[tuple[str, str]],
    r_value: int,
    hash_size: int,
    ngram_size: int
) -> dict:
    total_ham_matches = 0
    ham_message_count = 0
    total_spam_matches = 0
    spam_message_count = 0
    
    print("\n--- Calculating Detector Validation Statistics ---")
    
    for message, label in validation_set:
        encoded_msg = encode_message(message, hash_size, ngram_size)
        match_count = get_match_count(encoded_msg, detectors, r_value)
        
        if label == 'ham':
            ham_message_count += 1
            total_ham_matches += match_count
        elif label == 'spam':
            spam_message_count += 1
            total_spam_matches += match_count
    
    avg_ham = total_ham_matches / ham_message_count if ham_message_count > 0 else 0
    avg_spam = total_spam_matches / spam_message_count if spam_message_count > 0 else 0
    
    return {
        "average_matches_per_ham_message": round(avg_ham, 4),
        "average_matches_per_spam_message": round(avg_spam, 4)
    }

def evaluate_with_thresholds(
    test_results: list[tuple[int, str]],
    thresholds: list[int]
) -> dict[str, Any]:
    # Calculates performance metrics for a range of thresholds.

    results_by_threshold = {}
    print("\n--- Evaluation Results by Threshold ---")
    print("Threshold | F1-Score | Precision | Recall   | FP Rate")
    print("----------|----------|-----------|----------|----------")

    # loop through each threshold we want to test
    for threshold in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        # generate predictions based on the current threshold
        for match_count, actual_label in test_results:
            predicted_label = 'spam' if match_count >= threshold else 'ham'
            
            if actual_label == 'spam' and predicted_label == 'spam': tp += 1
            elif actual_label == 'ham' and predicted_label == 'spam': fp += 1
            elif actual_label == 'ham' and predicted_label == 'ham': tn += 1
            elif actual_label == 'spam' and predicted_label == 'ham': fn += 1

        # calculate metrics for this threshold
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # store results for this threshold
        results_by_threshold[f"threshold_{threshold}"] = {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "spam_metrics": {"precision": precision, "recall": recall, "f1_score": f1_score},
            "ham_metrics": {"false_positive_rate": fp_rate}
        }
        # print a summary row to the console
        print(f"    {threshold:<5} |   {f1_score:.4f} |    {precision:.4f} |   {recall:.4f} |   {fp_rate:.4f}")
    
    return results_by_threshold  
 