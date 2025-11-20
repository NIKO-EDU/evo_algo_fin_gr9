# ACIT 4610 Final Project - Group 9

This repository contains implementations of various optimization algorithms and machine learning techniques for solving different computational problems. The project includes four main problem sets, each demonstrating different algorithmic approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Problem 1: Ant Colony Optimization for Bin Packing](#problem-1-ant-colony-optimization-for-bin-packing)
- [Problem 3: Bees Algorithm for Knapsack Problem](#problem-3-bees-algorithm-for-knapsack-problem)
- [Problem 4: Negative Selection Algorithm for Spam Detection](#problem-4-negative-selection-algorithm-for-spam-detection)
- [Problem 5: Q-Learning for FrozenLake](#problem-5-q-learning-for-frozenlake)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

## 🎯 Overview

This project implements and compares various metaheuristic and machine learning algorithms:

1. **Ant Colony Optimization (ACO)** - Solves the one-dimensional bin packing problem
2. **Bees Algorithm** - Solves the 0/1 knapsack problem
3. **Negative Selection Algorithm (NSA)** - Detects spam messages using immune system-inspired detection
4. **Q-Learning** - Trains an agent to solve the FrozenLake environment using reinforcement learning

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd final_project_ACIT4610_group_9
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Problem 1: Ant Colony Optimization for Bin Packing

Implements Ant Colony Optimization to solve the one-dimensional bin packing problem, minimizing the number of bins needed to pack items of various sizes.

### Quick Start

```bash
cd problem-1

# Run with default test instance
python main.py

# Run with custom items
python main.py STANDARD --items "20,15,10,25,30" --capacity 50

# Run on benchmark file
python main.py STANDARD --instance data/binpack1.txt

# Run all benchmark experiments
python run_experiments.py --benchmarks
```

### Analyze Results

```bash
# Analyze and generate plots
python analysis.py STANDARD --instance simple_test

# For benchmark results
python analysis.py STANDARD --instance u120_00
```

### Configuration

Modify ACO parameters in `config.py`:
- `n_ants`: Number of ants per iteration (default: 10)
- `n_iterations`: Number of iterations (default: 1000)
- `alpha`: Pheromone importance (default: 1.0)
- `beta`: Heuristic importance (default: 2.0)
- `rho`: Evaporation rate (default: 0.1)

### Output

- **Data files**: Saved in `data/` directory (`.npy` files)
- **Plots**: Generated in `plots/` directory
  - Convergence plots showing algorithm progress
  - Load distribution plots showing box utilization

### Benchmark Data

OR-Library benchmark files (`binpack1.txt` - `binpack8.txt`) should be placed in the `data/` directory.

Download from: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html

---

## 📦 Problem 3: Bees Algorithm for Knapsack Problem

Implements the Bees Algorithm to solve the 0/1 knapsack problem using benchmark instances from the Pisinger dataset.

### Quick Start

```bash
cd problem-3

# Run a single problem instance
python main.py

# Run all problem instances
python runall.py

# Run statistical analysis (10 independent trials)
python run_10_independent.py
```

### Algorithm Parameters

The Bees Algorithm uses adaptive parameters based on problem size:
- **Small problems (≤50 items)**: 50 scout bees, 1000 iterations
- **Medium problems (≤100 items)**: 60 scout bees, 500 iterations
- **Large problems (≤500 items)**: 80 scout bees, 300 iterations
- **Very large problems (>500 items)**: 100 scout bees, 200 iterations

### Output

All plots are saved to the `plots/` directory:
- `convergence_*.png` - Fitness improvement over iterations
- `comparison_*.png` - Bar charts comparing Greedy vs Bees Algorithm

### Statistical Analysis

The `run_10_independent.py` script runs each instance 10 times with different random seeds and reports:
- Best, Worst, Mean, and Standard Deviation
- LaTeX formatted table for reports
- Takes approximately 30-60 minutes to complete

---

## 📦 Problem 4: Negative Selection Algorithm for Spam Detection

Implements the Negative Selection Algorithm (NSA), an immune system-inspired approach for detecting spam messages. The algorithm generates detectors that match spam patterns while avoiding matches with legitimate (ham) messages.

### Quick Start

```bash
cd problem-4

# Run the main experiment
python run.py

# Run baseline comparison (scikit-learn Naive Bayes)
python baseline.py
```

### Configuration

Modify parameters in `config.py`:
- `HASH_SIZE`: Size of binary hash vector (default: 512)
- `NGRAM_SIZE`: Size of n-grams for encoding (default: 3)
- `NUM_DETECTORS`: Number of detectors to generate (default: 1000)
- `R_CONTIGUOUS_VALUES`: List of r-contiguous matching thresholds (default: [13])
- `THRESHOLDS`: Detection thresholds for evaluation (default: [1, 2, 3, 5, 8, 10])

### How It Works

1. **Training Phase**: Generates detectors that don't match any ham messages in the training set
2. **Evaluation Phase**: Tests detectors on spam/ham messages and counts matches
3. **Classification**: Messages with match counts above a threshold are classified as spam

### Output

- **Detectors**: Saved in `data/detectors/` (`.npy` files)
- **Results**: Saved in `data/results/` (`.json` files)
  - Training time and acceptance rate
  - Detector validation statistics
  - Performance metrics (precision, recall, F1-score) for each threshold

### Dataset

The SMS Spam Collection dataset should be located at:
```
data/dataset/SMSSpamCollection
```

The dataset format: `label\tmessage` (tab-separated)

---

## 📦 Problem 5: Q-Learning for FrozenLake

Implements Q-Learning, a reinforcement learning algorithm, to train an agent to solve the FrozenLake environment from Gymnasium.

### Quick Start

```bash
cd problem-5

# Train a single experiment
python main.py 4x4_BASIC

# Train all experiments
python run_all.py

# Analyze results
python analysis.py 4x4_BASIC

# Visualize grid
python visualize_grid.py 4x4_BASIC
```

### Available Experiments

- `4x4_BASIC` - Standard 4x4 grid with basic parameters
- `8x8_HARD` - Larger 8x8 grid with extended training
- `4x4_MYOPIC` - 4x4 grid with low discount factor (short-sighted)
- `4x4_CAUTIOUS` - 4x4 grid with low learning rate (cautious learning)

### Configuration

Modify experiment parameters in `config.py`:
- `num_episodes`: Number of training episodes
- `learning_rate`: Q-value update rate (alpha)
- `discount_factor`: Future reward importance (gamma)
- `epsilon`: Initial exploration rate
- `epsilon_decay_rate`: Rate of exploration decay
- `min_epsilon`: Minimum exploration rate

### Output

- **Q-tables**: Saved in `data/` directory (`.npy` files)
- **Plots**: Generated in `plots/` directory
  - Learning curves showing success rate over episodes
  - Comparison with random and heuristic baselines
  - Epsilon decay visualization

### Baselines

The implementation includes two baselines for comparison:
- **Random Policy**: Random action selection
- **Heuristic Policy**: Simple heuristic moving toward goal (Down/Right preference)

---

## 📁 Project Structure

```
final_project_ACIT4610_group_9/
├── problem-1/          # ACO Bin Packing
│   ├── aco_bin_packing.py
│   ├── main.py
│   ├── analysis.py
│   ├── config.py
│   ├── data/
│   └── plots/
├── problem-3/          # Bees Algorithm Knapsack
│   ├── basolver.py
│   ├── knapsack.py
│   ├── main.py
│   ├── runall.py
│   ├── problems/
│   └── plots/
├── problem-4/          # NSA Spam Detection
│   ├── nsa.py
│   ├── encoder.py
│   ├── run.py
│   ├── evaluation.py
│   ├── data/
│   └── plots/
├── problem-5/          # Q-Learning FrozenLake
│   ├── main.py
│   ├── analysis.py
│   ├── baselines.py
│   ├── config.py
│   ├── data/
│   └── plots/
├── requirements.txt
└── README.md
```

## 📋 Requirements

All dependencies are listed in `requirements.txt`. Key packages include:

- **numpy** - Numerical computations
- **matplotlib** - Plotting and visualization
- **gymnasium** - Reinforcement learning environments
- **scikit-learn** - Machine learning utilities (for baseline comparisons)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Common Issues

### Import Errors
- Ensure you're in the correct directory when running scripts
- Verify all dependencies are installed: `pip install -r requirements.txt`

### File Not Found Errors
- Check that data files are in the correct directories
- For Problem 1: Ensure benchmark files are in `problem-1/data/`
- For Problem 3: Ensure problem files are in `problem-3/problems/`
- For Problem 4: Ensure dataset is in `problem-4/data/dataset/`

### Memory Issues
- Large problem instances may require significant memory
- Reduce `NUM_DETECTORS` in Problem 4 if detector generation fails
- Reduce `n_ants` or `n_iterations` in Problem 1 for very large instances

## 📊 Results and Analysis

Each problem directory contains:
- **Data files**: Saved results, Q-tables, detectors, etc.
- **Plots**: Visualization of algorithm performance
- **Configuration files**: Parameters used for experiments

Results can be analyzed using the provided analysis scripts in each problem directory.

## 📝 Notes

- All experiments use fixed random seeds for reproducibility
- Results are saved automatically for later analysis
- Plots are generated automatically after running experiments
- Each problem can be run independently

## 🤝 Contributing

This is a course project. For questions or issues, please refer to the individual problem README files or contact the project maintainers.

## 📄 License

This project is for educational purposes as part of ACIT 4610 coursework.

---

**Last Updated**: 11.20.2025
