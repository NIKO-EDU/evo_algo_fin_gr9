# run_experiments.py
# Utility script to run all experiments and save results

import subprocess
import sys
import os

# Get the python executable's path to be robust (works in virtual environments)
PYTHON_EXEC = sys.executable

# --- List of all experiments to run ---
EXPERIMENTS = [
    "STANDARD"
]

# Benchmark instances to test (at least 8 for the assignment)
# Format: (file_path, problem_index, description)
# Each binpack file contains 20 problems, we select diverse ones
BENCHMARK_INSTANCES = [
    ("data/binpack1.txt", 0, "u120_00 - 120 items, capacity 150 (uniform)"),
    ("data/binpack1.txt", 5, "u120_05 - 120 items, capacity 150 (uniform)"),
    ("data/binpack2.txt", 0, "u250_00 - 250 items, capacity 150 (uniform)"),
    ("data/binpack2.txt", 10, "u250_10 - 250 items, capacity 150 (uniform)"),
    ("data/binpack3.txt", 0, "u500_00 - 500 items, capacity 150 (uniform)"),
    ("data/binpack5.txt", 0, "t60_00 - 60 items, capacity 100 (triplets)"),
    ("data/binpack6.txt", 0, "t120_00 - 120 items, capacity 100 (triplets)"),
    ("data/binpack7.txt", 0, "t249_00 - 249 items, capacity 100 (triplets)"),
]

# Test instance (for quick testing without benchmark files)
TEST_INSTANCE = {
    "items": "20,15,10,25,30,12,18,22,8,14",
    "capacity": 50
}


def run_command(command_str: str):
    """Prints and runs a single command string in the terminal."""
    print(f"\n> Executing: {command_str}")
    # Using shell=True to simply run the command string as is.
    # For this controlled use case, it's safe and simple.
    result = subprocess.run(command_str, shell=True)
    if result.returncode != 0:
        print(f"Warning: Command failed with exit code {result.returncode}")
    return result.returncode == 0


# --- THE SCRIPT'S MAIN LOGIC ---

def run_benchmark_experiments():
    """Run experiments on OR-Library benchmark instances."""
    print("="*60)
    print("--- RUNNING BENCHMARK EXPERIMENTS ---")
    print("="*60)
    print(f"\nTesting {len(BENCHMARK_INSTANCES)} benchmark instances")
    print(f"Using experiment configuration: STANDARD")
    print()
    
    # Phase 1: Run all experiments
    print("\n--- Phase 1: Running ACO on All Benchmark Instances ---")
    for file_path, problem_index, description in BENCHMARK_INSTANCES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Instance: {description}")
        print(f"File: {file_path}, Problem index: {problem_index}")
        print(f"{'='*60}")
        
        # Run experiment with specified problem index
        command = f"{PYTHON_EXEC} main.py STANDARD --instance {file_path} --problem-index {problem_index}"
        success = run_command(command)
        if not success:
            print(f"Failed to run: {description}")
    
    # Phase 2: Analyze all results and generate plots
    print("\n" + "="*60)
    print("--- Phase 2: Analyzing Results and Generating Plots ---")
    print("="*60)
    
    # Import here to avoid circular imports
    from data_loader import load_orlib_instance
    
    for file_path, problem_index, description in BENCHMARK_INSTANCES:
        if not os.path.exists(file_path):
            continue
        
        try:
            # Load instance to get item_sizes and capacity for load distribution plot
            item_sizes, box_capacity, instance_name = load_orlib_instance(file_path, problem_index)
            
            print(f"\nAnalyzing: {instance_name} ({description})")
            
            # Run analysis (this generates plots and prints metrics)
            # Note: We need to pass items and capacity for load distribution plot
            items_str = ",".join(map(str, item_sizes))
            command = (
                f"{PYTHON_EXEC} analysis.py STANDARD "
                f"--instance {instance_name} "
                f"--items {items_str} "
                f"--capacity {box_capacity}"
            )
            success = run_command(command)
            if not success:
                print(f"Failed to analyze: {instance_name}")
        except Exception as e:
            print(f"Error analyzing {description}: {e}")


def run_test_experiments():
    """Run experiments on simple test instance."""
    print("="*60)
    print("--- STARTING FULL ACO BIN PACKING EXPERIMENT PIPELINE ---")
    print("="*60)

    # --- 1. RUN ALL EXPERIMENTS ---
    print("\n--- Phase 1: Running All ACO Experiments ---")
    for exp_name in EXPERIMENTS:
        command = (
            f"{PYTHON_EXEC} main.py {exp_name} "
            f"--items {TEST_INSTANCE['items']} "
            f"--capacity {TEST_INSTANCE['capacity']}"
        )
        success = run_command(command)
        if not success:
            print(f"Failed to run experiment: {exp_name}")

    # --- 2. ANALYZE ALL RESULTS ---
    print("\n--- Phase 2: Analyzing All Results ---")
    for exp_name in EXPERIMENTS:
        command = (
            f"{PYTHON_EXEC} analysis.py {exp_name} "
            f"--items {TEST_INSTANCE['items']} "
            f"--capacity {TEST_INSTANCE['capacity']}"
        )
        success = run_command(command)
        if not success:
            print(f"Failed to analyze experiment: {exp_name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ACO bin packing experiments")
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run experiments on OR-Library benchmark instances (default: run test experiments)"
    )
    args = parser.parse_args()
    
    if args.benchmarks:
        run_benchmark_experiments()
    else:
        run_test_experiments()
    
    print("\n" + "="*60)
    print("--- PIPELINE COMPLETE ---")
    print("="*60)
    print("\nResults saved in 'data/' directory")
    print("Plots saved in 'plots/' directory")

