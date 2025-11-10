# baseline.py

import os
import json
import time # Import the time module
from typing import Any
# Scikit-learn is the only dependency you might need to install:
# pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Project Imports ---
# Import the shared data loader function and the seed for reproducibility.
from data_loader import load_sms_data
from config import SEED

# --- Configuration ---
# The path is hardcoded here as requested, matching the pattern in run.py.
DATASET_PATH = os.path.join('data', 'dataset', 'SMSSpamCollection')
RESULTS_DIR = os.path.join('data', 'results')
# The filename now dynamically includes the SEED value.
RESULTS_FILENAME = f'results_baseline_S{SEED}.json'

# --- Main Execution ---
def main():
    # This script trains a baseline model and saves its evaluation metrics to a JSON file.
    
    print("===== RUNNING BASELINE MODEL EXPERIMENT =====")

    # 1. Load the data
    print(f"Loading data from '{DATASET_PATH}'...")
    ham_messages, spam_messages = load_sms_data(DATASET_PATH)
    if not ham_messages or not spam_messages:
        print("Could not load data. Exiting.")
        return

    # 2. Prepare data for scikit-learn
    all_messages = ham_messages + spam_messages
    all_labels = [0] * len(ham_messages) + [1] * len(spam_messages)

    # 3. Create a reproducible Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        all_messages,
        all_labels,
        test_size=0.3,
        random_state=SEED,
        stratify=all_labels
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 4. Define the Model Pipeline
    print("Building model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])

    # 5. Train the Model (with timing)
    print("Training the baseline model...")
    start_train_time = time.time()
    pipeline.fit(X_train, y_train)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    print(f"Training complete in {training_duration:.4f} seconds.")

    # 6. Evaluate the Model (with timing)
    print("\n--- Evaluating Model Performance on the Test Set ---")
    start_eval_time = time.time()
    y_pred = pipeline.predict(X_test)
    end_eval_time = time.time()
    evaluation_duration = end_eval_time - start_eval_time
    print(f"Evaluation complete in {evaluation_duration:.4f} seconds.")

    # Get the classification report and confusion matrix
    report_dict = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
    assert isinstance(report_dict, dict)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results to the console as before
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print("Confusion Matrix:")
    print(cm)
    
    # 7. Save the Results to a JSON file
    results_to_save = {
        "model_name": "Baseline - Multinomial Naive Bayes with TF-IDF",
        "seed": SEED,
        "timing_info": {
            "training_time_seconds": round(training_duration, 4),
            "evaluation_time_seconds": round(evaluation_duration, 4)
        },
        "evaluation_metrics": {
            "spam_metrics": report_dict['spam'],
            "accuracy": report_dict["accuracy"]
        },
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        }
    }

    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
    
    # Write the results to the file
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=4)
        
    print(f"\nEvaluation results successfully saved to '{filepath}'")
    print("\n===== BASELINE EXPERIMENT COMPLETE =====")

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path is not found or is incorrect: '{DATASET_PATH}'")
    else:
        main()