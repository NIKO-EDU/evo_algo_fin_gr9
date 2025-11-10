# file_utils.py

import os
import json
import numpy as np
from typing import Any

def save_detectors(
    detectors: list[np.ndarray],
    r_value: int,
    num_detectors: int,
    seed: int
):
    # Saves the list of detector arrays to a .npy file.
    
    # create the directory if it doesn't exist
    detector_dir = os.path.join('data', 'detectors')
    os.makedirs(detector_dir, exist_ok=True)
    
    # filename 
    filename = f"detectors_S{seed}_R{r_value}_N{num_detectors}.npy"
    filepath = os.path.join(detector_dir, filename)
    
    # convert list of arrays to a matrix and save
    detector_array = np.array(detectors)
    np.save(filepath, detector_array)
    print(f"Detectors successfully saved to '{filepath}'")

def load_detectors(
    r_value: int,
    num_detectors: int,
    seed: int
) -> list[np.ndarray] | None:
    
    detector_dir = os.path.join('data', 'detectors')
    filename = f"detectors_S{seed}_R{r_value}_N{num_detectors}.npy"
    filepath = os.path.join(detector_dir, filename)
    
    # guard clause to check if the file exists
    if not os.path.exists(filepath):
        print(f"Detector file not found at '{filepath}'. Need to generate them first.")
        return None
    
    # load the matrix and conver to list 
    loaded_array = np.load(filepath)
    detectors = [row for row in loaded_array]
    
    print(f"Successfully loaded {len(detectors)} detectors from '{filepath}'")
    return detectors

def save_results(
    results_data: dict[str, Any]
):
    # Saves the evaluation results
    
    # create the directory 
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # get hyperparameters 
    params = results_data['hyperparameters']
    r_val = params['R_CONTIGUOUS']
    n_detect = params['NUM_DETECTORS']
    
    filename = f"results_R{r_val}_N{n_detect}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"Evaluation results successfully saved to '{filepath}'")