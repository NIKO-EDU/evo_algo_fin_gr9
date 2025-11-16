# analysis.py
# Analyze and visualize ACO bin packing results

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, Any, Optional

# Import our modules
import config
from data_loader import load_orlib_instance, create_simple_instance


def load_results(experiment_name: str, instance_name: str) -> Optional[Dict[str, Any]]:
    """
    Load saved experiment results from files.
    
    Loads the convergence history and solution data that were saved by main.py.
    These files should be in the 'data/' directory with naming pattern:
    - {experiment_name}_{instance_name}_convergence.npy
    - {experiment_name}_{instance_name}_solution.npy
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment configuration (e.g., "STANDARD")
    instance_name : str
        Name of the problem instance (e.g., "simple_test")
        
    Returns:
    --------
    Optional[Dict[str, Any]] : Dictionary with keys:
        - 'convergence': numpy array of best num_boxes per iteration
        - 'solution': dict with assignment, num_boxes, unused_capacity, box_loads, runtime
        Returns None if files not found
    """
    run_id = f"{experiment_name}_{instance_name}"
    
    convergence_path = os.path.join("data", f"{run_id}_convergence.npy")
    solution_path = os.path.join("data", f"{run_id}_solution.npy")
    
    if not os.path.exists(convergence_path) or not os.path.exists(solution_path):
        print(f"Error: Results not found for '{run_id}'.")
        print("Please run the training script first.")
        return None
    
    convergence = np.load(convergence_path)
    solution_data = np.load(solution_path, allow_pickle=True).item()
    
    return {
        'convergence': convergence,
        'solution': solution_data
    }


def plot_convergence(
    convergence: np.ndarray,
    experiment_name: str,
    instance_name: str
) -> None:
    """
    Plot convergence curve showing how ACO improves over iterations.
    
    Creates a line plot showing:
    - X-axis: iteration number
    - Y-axis: best number of boxes found so far
    
    This visualization helps understand:
    - How quickly ACO converges
    - If the algorithm is still improving or has converged
    
    Saved to: plots/{experiment_name}_{instance_name}_convergence.png
    
    Parameters:
    -----------
    convergence : np.ndarray
        Convergence history (best num_boxes per iteration)
    experiment_name : str
        Name of the experiment configuration
    instance_name : str
        Name of the problem instance
    """
    plt.figure(figsize=(12, 6))
    
    # Plot convergence
    plt.plot(convergence, label='ACO Best Solution', color='blue', linewidth=2)
    
    plt.title(f'Convergence: {experiment_name} - {instance_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Boxes (Best So Far)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    plot_path = os.path.join("plots", f"{experiment_name}_{instance_name}_convergence.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to {plot_path}")
    plt.close()


def plot_load_distribution(
    box_loads: list,
    box_capacity: int,
    experiment_name: str,
    instance_name: str
) -> None:
    """
    Plot load distribution showing how items are distributed across boxes.
    
    Creates a bar chart showing:
    - X-axis: box number (1, 2, 3, ...)
    - Y-axis: lo ad (sum of item sizes in that box)
    - Red dashed line: box capacity limit
    
    This visualization helps understand:
    - How efficiently boxes are filled
    - Whether some boxes are underutilized
    - Overall packing quality
    
    Saved to: plots/{experiment_name}_{instance_name}_load_distribution.png
    
    Parameters:
    -----------
    box_loads : list
        Load of each box (sum of item sizes in that box)
    box_capacity : int
        Maximum capacity of each box (shown as red line)
    experiment_name : str
        Name of the experiment configuration
    instance_name : str
        Name of the problem instance
    """
    num_boxes = len(box_loads)
    box_indices = range(1, num_boxes + 1)
    
    # Adjust figure size based on number of boxes
    if num_boxes > 100:
        figsize = (max(16, num_boxes * 0.15), 8)
    elif num_boxes > 50:
        figsize = (14, 7)
    else:
        figsize = (12, 6)
    
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(box_indices, box_loads, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add capacity line
    plt.axhline(
        y=box_capacity,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Capacity ({box_capacity})'
    )
    
    # Add value labels on bars only if not too many boxes
    # For many boxes, labels become unreadable
    if num_boxes <= 50:
        # Show all labels
        for i, (idx, load) in enumerate(zip(box_indices, box_loads)):
            plt.text(idx, load + box_capacity * 0.01, f'{load}', 
                    ha='center', va='bottom', fontsize=9)
    elif num_boxes <= 100:
        # Show every 5th label
        for i, (idx, load) in enumerate(zip(box_indices, box_loads)):
            if i % 5 == 0:
                plt.text(idx, load + box_capacity * 0.01, f'{load}', 
                        ha='center', va='bottom', fontsize=8)
    # For > 100 boxes, don't show labels (too cluttered)
    
    plt.title(f'Load Distribution: {experiment_name} - {instance_name}')
    plt.xlabel('Box Number')
    plt.ylabel('Load')
    
    # Adjust x-axis ticks based on number of boxes
    if num_boxes <= 20:
        plt.xticks(box_indices)
    elif num_boxes <= 50:
        plt.xticks(box_indices[::2])  # Every 2nd box
    elif num_boxes <= 100:
        plt.xticks(box_indices[::5])  # Every 5th box
    else:
        plt.xticks(box_indices[::10])  # Every 10th box
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, box_capacity * 1.1)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    plot_path = os.path.join("plots", f"{experiment_name}_{instance_name}_load_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Load distribution plot saved to {plot_path}")
    plt.close()


def main() -> None:
    """
    Main function to analyze ACO bin packing results.
    
    This function:
    1. Loads saved experiment results from files
    2. Prints ACO results
    3. Generates convergence plot
    4. Generates load distribution plot
    
    Usage examples:
        python3 analysis.py STANDARD --instance simple_test --items "20,15,10,25,30" --capacity 50
        python3 analysis.py STANDARD --instance u120_00
    """
    parser = argparse.ArgumentParser(description="Analyze ACO bin packing results.")
    parser.add_argument(
        "experiment_name",
        type=str,
        help="The name of the experiment to analyze (e.g., STANDARD)."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="simple_test",
        help="Instance name (default: simple_test)."
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        help="Comma-separated item sizes (if recreating instance)."
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=50,
        help="Box capacity (if recreating instance)."
    )
    args = parser.parse_args()
    
    exp_name = args.experiment_name
    
    if exp_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{exp_name}' not found in config.py.")
        return
    
    print(f"--- Analyzing Experiment: {exp_name} ---")
    
    # --- Load Results ---
    results = load_results(exp_name, args.instance)
    if results is None:
        return
    
    convergence = results['convergence']
    solution = results['solution']
    
    # --- Recreate Instance for Load Distribution Plot ---
    if args.items:
        item_sizes = [int(x.strip()) for x in args.items.split(',')]
        box_capacity = args.capacity
    else:
        # Try to infer from solution (this is a limitation - we'd need to save item_sizes)
        # For now, use default test instance
        item_sizes = [20, 15, 10, 25, 30, 12, 18, 22, 8, 14]
        box_capacity = 50
    
    # Print results
    print("\n" + "="*60)
    print("ACO RESULTS")
    print("="*60)
    print(f"Number of boxes: {solution['num_boxes']}")
    print(f"Unused capacity: {solution['unused_capacity']}")
    print(f"Runtime: {solution['runtime']:.2f} seconds")
    
    # --- Generate Plots ---
    print("\nGenerating plots...")
    plot_convergence(convergence, exp_name, args.instance)
    plot_load_distribution(
        solution['box_loads'],
        box_capacity,
        exp_name,
        args.instance
    )
    
    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    main()

