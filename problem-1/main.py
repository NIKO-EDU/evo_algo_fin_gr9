# main.py
# Main script to run ACO bin packing experiments

import argparse
import time
import os
from typing import Any
import numpy as np

# Import our modules
import config
from aco_bin_packing import ACOBinPacking
from baselines import first_fit_decreasing
from data_loader import load_orlib_instance, create_simple_instance, get_benchmark_instances


def save_results(
    experiment_name: str,
    instance_name: str,
    result: dict,
    runtime: float
) -> None:
    """
    Save experiment results to files for later analysis.
    
    Saves two files:
    1. Convergence history: array of best num_boxes per iteration (for plotting)
    2. Solution details: assignment, num_boxes, unused_capacity, box_loads, runtime
    
    Files are saved in the 'data/' directory with naming pattern:
    - {experiment_name}_{instance_name}_convergence.npy
    - {experiment_name}_{instance_name}_solution.npy
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment configuration (e.g., "STANDARD")
    instance_name : str
        Name of the problem instance (e.g., "simple_test", "binpack1")
    result : dict
        Result dictionary from ACO solver containing:
        - 'assignment': item to box mapping
        - 'num_boxes': number of boxes used
        - 'unused_capacity': total unused capacity
        - 'box_loads': list of loads for each box
        - 'convergence': list of best num_boxes per iteration
    runtime : float
        Runtime in seconds (added to solution data)
    """
    print("Saving results...")
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Create a unique identifier for this run
    run_id = f"{experiment_name}_{instance_name}"
    
    # Save convergence history
    convergence_path = os.path.join("data", f"{run_id}_convergence.npy")
    np.save(convergence_path, np.array(result['convergence']))
    
    # Save solution details
    solution_path = os.path.join("data", f"{run_id}_solution.npy")
    solution_data = {
        'assignment': result['assignment'],
        'num_boxes': result['num_boxes'],
        'unused_capacity': result['unused_capacity'],
        'box_loads': result['box_loads'],
        'runtime': runtime,
        'baseline': result.get('baseline')
    }
    np.save(solution_path, solution_data, allow_pickle=True)
    
    print(f"Results saved to {convergence_path} and {solution_path}")


def main() -> None:
    """
    Main function to run ACO bin packing experiments.
    
    This is the entry point for running experiments. It:
    1. Parses command-line arguments (experiment name, instance, etc.)
    2. Loads experiment configuration from config.py
    3. Loads or creates problem instance
    4. Initializes ACO solver with configuration parameters
    5. Runs ACO algorithm to solve the problem
    6. Saves results to files for later analysis
    
    Usage examples:
        python3 main.py STANDARD --items "20,15,10,25,30" --capacity 50
        python3 main.py STANDARD --instance data/binpack1.txt
        python main.py STANDARD  # (uses default test instance)
    """
    # --- Step 0: Load Configuration FROM COMMAND LINE ---
    parser = argparse.ArgumentParser(
        description="Run ACO bin packing experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                    # Run with STANDARD config and test instance
  python3 main.py STANDARD           # Same as above
  python3 main.py STANDARD --items "20,15,10" --capacity 50
  python3 main.py STANDARD --instance data/binpack1.txt
        """
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        nargs='?',  # Make optional
        default="STANDARD",
        help="The name of the experiment configuration (default: STANDARD)."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Path to instance file (optional, uses simple test instance if not provided)."
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="Problem index to load from file (default: 0, for OR-Library files with multiple problems)."
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        help="Comma-separated item sizes (e.g., '20,15,10,25,30')."
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=50,
        help="Box capacity (default: 50)."
    )
    args = parser.parse_args()
    
    exp_name = args.experiment_name
    
    if exp_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{exp_name}' not found in config.py.")
        print(f"Available experiments: {list(config.EXPERIMENTS.keys())}")
        return
    
    exp_config: dict[str, Any] = config.EXPERIMENTS[exp_name]
    print(f"--- Running Experiment: {exp_config['name']} ---")
    
    # --- Step 1: Load Problem Instance ---
    if args.instance:
        # Load from file
        try:
            # Check if it's an OR-Library file (contains multiple problems)
            from data_loader import list_problems_in_file
            problems = list_problems_in_file(args.instance)
            if len(problems) > 1:
                print(f"File contains {len(problems)} problems:")
                for idx, prob_id, cap, n_items in problems[:5]:  # Show first 5
                    print(f"  [{idx}] {prob_id}: {n_items} items, capacity {cap}")
                if len(problems) > 5:
                    print(f"  ... and {len(problems) - 5} more")
                print(f"\nLoading first problem (index 0): {problems[0][1]}")
            
            item_sizes, box_capacity, instance_name = load_orlib_instance(args.instance, problem_index=args.problem_index)
            print(f"Loaded instance: {instance_name}")
        except Exception as e:
            print(f"Error loading instance: {e}")
            return
    elif args.items:
        # Create from command line arguments
        item_sizes = [int(x.strip()) for x in args.items.split(',')]
        box_capacity = args.capacity
        instance_name = "command_line"
        print(f"Using command-line instance: {len(item_sizes)} items, capacity {box_capacity}")
    else:
        # Use simple test instance
        item_sizes = [20, 15, 10, 25, 30, 12, 18, 22, 8, 14]
        box_capacity = 50
        instance_name = "simple_test"
        print(f"Using simple test instance: {len(item_sizes)} items, capacity {box_capacity}")
    
    print(f"Items: {item_sizes}")
    print(f"Box capacity: {box_capacity}")
    print()

    # --- Baseline: First-Fit Decreasing ---
    print("--- Baseline: First-Fit Decreasing (FFD) ---")
    baseline_start = time.time()
    baseline_result = first_fit_decreasing(item_sizes, box_capacity)
    baseline_runtime = time.time() - baseline_start
    print(f"Number of boxes (FFD): {baseline_result['num_boxes']}")
    print(f"Unused capacity (FFD): {baseline_result['unused_capacity']}")
    print(f"Runtime (FFD): {baseline_runtime:.4f} seconds")
    print()
    
    # --- Step 2: Initialize ACO Solver ---
    solver = ACOBinPacking(
        item_sizes=item_sizes,
        box_capacity=box_capacity,
        n_ants=int(exp_config["n_ants"]),
        n_iterations=int(exp_config["n_iterations"]),
        alpha=float(exp_config["alpha"]),
        beta=float(exp_config["beta"]),
        rho=float(exp_config["rho"]),
        Q=float(exp_config["Q"]),
        tau_init=float(exp_config["tau_init"]),
        seed=int(exp_config["seed"])
    )
    
    # --- Step 3: Solve ---
    start_time = time.time()
    result = solver.solve()
    runtime = time.time() - start_time
    
    print(f"\n--- Solution for {exp_config['name']} ---")
    print(f"Number of boxes used: {result['num_boxes']}")
    print(f"Unused capacity: {result['unused_capacity']}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"\nBox loads: {result['box_loads']}")

    # Attach baseline results for downstream analysis
    result['baseline'] = {
        'assignment': baseline_result['assignment'],
        'box_loads': baseline_result['box_loads'],
        'num_boxes': baseline_result['num_boxes'],
        'unused_capacity': baseline_result['unused_capacity'],
        'runtime': baseline_runtime
    }
    
    # --- Step 4: Save Results ---
    save_results(exp_name, instance_name, result, runtime)
    
    print(f"\n--- Experiment complete ---")


if __name__ == "__main__":
    main()

