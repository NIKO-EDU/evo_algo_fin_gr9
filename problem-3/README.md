# Problem 3: Bees Algorithm for 0/1 Knapsack Problem

This project implements the Bees Algorithm to solve the 0/1 Knapsack Problem using benchmark instances from the Pisinger dataset.

## Requirements

- Python 3.9 or higher
- matplotlib (for visualization)
- numpy (for numerical operations)

Install dependencies:
```bash
pip install matplotlib numpy
```

## Files

- `knapsack.py` - Core knapsack problem classes and repair function
- `baseline.py` - Greedy baseline algorithm
- `basolver.py` - Bees Algorithm implementation
- `plotting.py` - Visualization functions
- `main.py` - Run single instance
- `runall.py` - Run all instances sequentially
- `run_10_independent.py` - Run 10 trials per instance with different seeds
- `problems/` - CSV files with knapsack instances
- `plots/` - Generated convergence and comparison plots

## Usage

### Run a Single Problem

```bash
python3 main.py
```

This will solve one knapsack instance and generate plots in the `plots/` folder.

### Run All Problems

```bash
python3 runall.py
```

This runs all 7 problem instances (20, 50, 100, 200, 500, 1000, 2000 items) and generates:
- Convergence plots showing algorithm progress
- Comparison plots between Greedy baseline and Bees Algorithm
- Summary statistics in the terminal

### Run Statistical Analysis (10 Independent Trials)

```bash
python3 run_10_independent.py
```

This runs each instance 10 times with different random seeds (0-9) and reports:
- Best, Worst, Mean, and Standard Deviation
- LaTeX formatted table for reports
- Takes approximately 30-60 minutes to complete

## Algorithm Parameters

The Bees Algorithm uses the following parameters:
- **ns** (scout bees): 50
- **nre** (elite sites): 5
- **nrb** (best sites): 15
- **nbe** (elite neighborhood): 10
- **nbb** (best neighborhood): 5
- **Iterations**: Adaptive based on problem size (200-1000)

## Output

All plots are saved to the `plots/` directory:
- `convergence_*.png` - Shows fitness improvement over iterations
- `comparison_*.png` - Bar charts comparing Greedy vs Bees Algorithm

## Results

The implementation correctly solves small to medium instances (up to 200 items), matching the Greedy baseline. For larger instances (500+ items), the algorithm shows performance degradation compared to the baseline, highlighting the need for parameter tuning and more iterations on large-scale problems.

See `main.tex` for detailed analysis and discussion of results.
