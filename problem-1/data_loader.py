# data_loader.py
# Load benchmark instances for bin packing problem

from typing import Tuple, List, Optional
import os


def load_orlib_instance(file_path: str, problem_index: int = 0) -> Tuple[List[int], int, str]:
    """
    Load a bin packing instance from OR-Library format.
    
    OR-Library binpack files contain MULTIPLE problems. Each file has:
    - Line 1: Number of test problems (P)
    - For each problem:
      - Problem identifier (e.g., "u120_01")
      - Bin capacity, Number of items (n), Number of bins in best solution
      - n lines with item sizes
    
    This function loads ONE problem from the file.
    
    Parameters:
    -----------
    file_path : str
        Path to the instance file (e.g., "data/binpack1.txt")
    problem_index : int
        Which problem to load from the file (0 = first problem, default: 0)
        
    Returns:
    --------
    Tuple[List[int], int, str]
        (item_sizes, box_capacity, instance_name)
        - item_sizes: list of item sizes
        - box_capacity: maximum capacity per box
        - instance_name: problem identifier from file (e.g., "u120_01")
        
    Raises:
    -------
    FileNotFoundError: if file doesn't exist
    ValueError: if file format is invalid or problem_index is out of range
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instance file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) < 2:
        raise ValueError(f"Invalid instance file format: {file_path}")
    
    # First line: number of problems
    num_problems = int(lines[0])
    
    if problem_index < 0 or problem_index >= num_problems:
        raise ValueError(
            f"Problem index {problem_index} out of range. "
            f"File contains {num_problems} problems (indices 0-{num_problems-1})"
        )
    
    # Find the start of the requested problem
    line_idx = 1
    for p in range(problem_index):
        # Skip problem identifier
        line_idx += 1
        # Parse: capacity, n_items, best_bins
        parts = lines[line_idx].split()
        n_items = int(parts[1])
        line_idx += 1 + n_items  # Skip capacity line and all item sizes
    
    # Now at the requested problem
    problem_id = lines[line_idx]  # Problem identifier
    line_idx += 1
    
    # Parse: capacity, n_items, best_bins
    parts = lines[line_idx].split()
    # Handle both integer and float capacities (binpack1-4 are int, binpack5-8 are float)
    box_capacity = int(float(parts[0]))  # Convert float to int if needed
    n_items = int(parts[1])
    line_idx += 1
    
    # Read item sizes (handle both int and float, convert to int)
    item_sizes = [int(float(lines[line_idx + i])) for i in range(n_items)]
    
    return item_sizes, box_capacity, problem_id


def list_problems_in_file(file_path: str) -> List[Tuple[int, str, int, int]]:
    """
    List all problems in an OR-Library binpack file.
    
    Parameters:
    -----------
    file_path : str
        Path to the instance file
        
    Returns:
    --------
    List[Tuple[int, str, int, int]]
        List of (index, problem_id, capacity, n_items) for each problem
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instance file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    num_problems = int(lines[0])
    problems = []
    line_idx = 1
    
    for p in range(num_problems):
        problem_id = lines[line_idx]
        line_idx += 1
        parts = lines[line_idx].split()
        # Handle both integer and float capacities (binpack1-4 are int, binpack5-8 are float)
        capacity = int(float(parts[0]))  # Convert float to int if needed
        n_items = int(parts[1])
        line_idx += 1 + n_items
        
        problems.append((p, problem_id, capacity, n_items))
    
    return problems


def create_simple_instance(item_sizes: List[int], box_capacity: int, name: str = "simple") -> Tuple[List[int], int, str]:
    """
    Create a simple test instance (for testing without benchmark files).
    
    This is a convenience function for creating problem instances programmatically
    without needing to create files. Useful for quick testing or when you want
    to specify items directly in code.
    
    Parameters:
    -----------
    item_sizes : List[int]
        Sizes of items to pack
    box_capacity : int
        Maximum capacity of each box
    name : str
        Name identifier for this instance (default: "simple")
        
    Returns:
    --------
    Tuple[List[int], int, str]
        (item_sizes, box_capacity, instance_name)
        Returns the same inputs in a standardized format
    """
    return item_sizes, box_capacity, name


def get_benchmark_instances(data_dir: str = "data") -> List[Tuple[str, str]]:
    """
    Get list of available benchmark instances.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing benchmark files
        
    Returns:
    --------
    List[Tuple[str, str]]
        List of (instance_name, file_path) tuples
    """
    instances = []
    
    if not os.path.exists(data_dir):
        return instances
    
    # Look for common OR-Library file patterns
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt') or filename.startswith('binpack'):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                instance_name = filename
                if instance_name.endswith('.txt'):
                    instance_name = instance_name[:-4]
                instances.append((instance_name, file_path))
    
    return sorted(instances)
