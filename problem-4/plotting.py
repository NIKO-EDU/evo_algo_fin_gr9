# plotting.py

import os
import json
import re
import glob
import matplotlib.pyplot as plt

# --- Configuration ---
RESULTS_DIR = os.path.join('data', 'results')
COVERAGE_DATA_FILE = 'coverage_data_T1_R14_N2000.json'
SAVE_DIR = 'plots'

def plot_precision_recall_curve():
    """
    Loads all NSA results files (results_Rxx_Nxxxx.json) and plots the 
    Precision-Recall curve for each R value based on the different thresholds 
    that were tested.
    """
    print("Generating Precision-Recall Curve...")
    
    # Find all results files matching the pattern results_Rxx_Nxxxx.json
    pattern = os.path.join(RESULTS_DIR, 'results_R*_N*.json')
    result_files = glob.glob(pattern)
    
    # Filter out baseline and speedtest files
    result_files = [f for f in result_files if 'baseline' not in f and 'speedtest' not in f]
    
    if not result_files:
        print(f"Error: No results files found matching pattern '{pattern}'")
        return
    
    # Sort files by R value for consistent ordering
    def extract_r_value(filepath):
        match = re.search(r'results_R(\d+)_N', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    result_files.sort(key=extract_r_value)
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # Define colors for different curves
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / max(len(result_files) - 1, 1)) for i in range(len(result_files))]
    
    # Process each results file
    for idx, filepath in enumerate(result_files):
        filename = os.path.basename(filepath)
        
        # Extract R value from filename for labeling
        r_match = re.search(r'results_R(\d+)_N', filename)
        r_value = r_match.group(1) if r_match else '?'
        
        try:
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

            # Plot the curve for this R value
            plt.plot(recalls, precisions, marker='o', linestyle='--', 
                    label=f'R={r_value}', linewidth=2, markersize=6, color=colors[idx])
            
            # Annotate each point with its threshold value
            for recall, precision, threshold in zip(recalls, precisions, threshold_labels):
                plt.annotate(threshold, (recall, precision), 
                            textcoords="offset points", 
                            xytext=(0, 8), 
                            ha='center', 
                            fontsize=7,
                            color=colors[idx])
            
            print(f"  Loaded data from {filename} (R={r_value})")
            
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")
            continue

    plt.title('Precision-Recall Curve for NSA Model (All R Values)')
    plt.xlabel('Recall (Spam)')
    plt.ylabel('Precision (Spam)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    
    # Save the plot to a file
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, 'precision_recall_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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