# plotting.py

import os
import json
import matplotlib.pyplot as plt

# --- Configuration ---
RESULTS_DIR = os.path.join('data', 'results')
NSA_RESULTS_FILE = 'results_R14_N2000.json'
COVERAGE_DATA_FILE = 'coverage_data.json'
SAVE_DIR = 'plots'

def plot_precision_recall_curve():
    """
    Loads NSA results and plots the Precision-Recall curve based on
    the different thresholds that were tested.
    """
    print("Generating Precision-Recall Curve...")
    
    filepath = os.path.join(RESULTS_DIR, NSA_RESULTS_FILE)
    if not os.path.exists(filepath):
        print(f"Error: Results file not found at '{filepath}'")
        return

    with open(filepath, 'r') as f:
        results = json.load(f)

    threshold_evals = results['evaluation_by_threshold']
    
    precisions = []
    recalls = []
    threshold_labels = []

    # Sort thresholds numerically to ensure the plot is drawn in order
    sorted_threshold_keys = sorted(threshold_evals.keys(), key=lambda t: int(t.split('_')[1]))

    for key in sorted_threshold_keys:
        threshold = key.split('_')[1]
        metrics = threshold_evals[key]['spam_metrics']
        
        if metrics['precision'] > 0 or metrics['recall'] > 0:
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            threshold_labels.append(threshold)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='--', color='b')

    # Annotate each point with its threshold value
    for i, label in enumerate(threshold_labels):
        plt.annotate(f"T={label}", (recalls[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Precision-Recall Curve for NSA Model (R=14)')
    plt.xlabel('Recall (Spam)')
    plt.ylabel('Precision (Spam)')
    plt.grid(True)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    # Save the plot to a file
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'precision_recall_curve.png')
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    
    plt.show()

def plot_detector_coverage_curve():
    """
    Loads the generated coverage data and plots recall vs. number of detectors.
    """
    print("\nGenerating Detector Coverage Curve...")
    
    filepath = os.path.join(RESULTS_DIR, COVERAGE_DATA_FILE)
    if not os.path.exists(filepath):
        print(f"Error: Coverage data file not found at '{filepath}'")
        print("Please run generate_coverage_data.py first.")
        return

    with open(filepath, 'r') as f:
        coverage_data = json.load(f)
    
    num_detectors = [item['num_detectors'] for item in coverage_data]
    recalls = [item['recall'] for item in coverage_data]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(num_detectors, recalls, marker='o', linestyle='-', color='g')
    
    plt.title('Detector Coverage (Recall vs. Number of Detectors)')
    plt.xlabel('Number of Detectors Used')
    plt.ylabel('Recall (Spam)')
    plt.grid(True)
    plt.ylim(0, 1.05)

    # Save the plot to a file
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'detector_coverage_curve.png')
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")

    plt.show()

if __name__ == '__main__':
    plot_precision_recall_curve()
    plot_detector_coverage_curve()