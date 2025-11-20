# ACO Bin Packing Solver

Ant Colony Optimization (ACO) implementation for One-Dimensional Bin Packing Problem.

## Quick Start

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Installation

1. **Activate virtual environment** (if using one):
   ```bash
   source ../venv/bin/activate
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r ../requirements.txt
   ```

### Running the Algorithm

#### Option 1: Run Single Experiment (Easiest)
```bash
python3 main.py
```
or
```bash
python3 main.py STANDARD
```
This uses a built-in test instance (no files needed).

#### Option 2: Run with Custom Items
```bash
python3 main.py STANDARD --items "20,15,10,25,30" --capacity 50
```

#### Option 3: Run on Benchmark Files
If you have downloaded OR-Library benchmark files (binpack1.txt - binpack8.txt):
```bash
python3 main.py STANDARD --instance data/binpack1.txt
```

#### Option 4: Run All Benchmark Experiments
```bash
python3 run_experiments.py --benchmarks
```

### Analyzing Results

After running experiments, analyze and generate plots:
```bash
python3 analysis.py STANDARD --instance simple_test
```

For benchmark results:
```bash
python3 analysis.py STANDARD --instance u120_00
```

## Project Structure

- `aco_bin_packing.py` - Core ACO algorithm implementation
- `main.py` - Main script to run experiments
- `analysis.py` - Analyze results and generate plots
- `config.py` - ACO parameter configurations
- `data_loader.py` - Load benchmark instances
- `run_experiments.py` - Batch run multiple experiments
- `data/` - Benchmark files and saved results
- `plots/` - Generated visualization plots

## Configuration

ACO parameters can be modified in `config.py`. The configuration includes:
- `n_ants`: Number of ants (solutions) per iteration
- `n_iterations`: Number of iterations to run
- `alpha`: Pheromone importance (exploitation)
- `beta`: Heuristic importance (exploration)
- `rho`: Evaporation rate
- `Q`: Pheromone deposit quantity

## Benchmark Data

OR-Library benchmark files (binpack1.txt - binpack8.txt) should be placed in the `data/` directory.

Download from: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html

## Output

Results are saved in:
- `data/` - Convergence history and solution data (.npy files)
- `plots/` - Convergence plots and load distribution plots (.png files)

## Example Output

When running, you'll see:
- Progress updates every 10 iterations
- Final solution: number of boxes used, unused capacity, runtime
- Results saved to files for later analysis

