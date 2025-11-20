# config.py
# Configuration file for ACO Bin Packing experiments

# A dictionary to hold all our experiment configurations
EXPERIMENTS = {
    "STANDARD": {
        "name": "Standard ACO Configuration",
        "n_ants": 10,
        "n_iterations": 1000,
        "alpha": 1.0,      # Pheromone importance (exploitation)
        "beta": 2.0,       # Heuristic importance (exploration)
        "rho": 0.1,        # Evaporation rate
        "Q": 1.0,          # Pheromone deposit quantity
        "tau_init": 0.1,   # Initial pheromone level
        "seed": 42
    }
}

# Default experiment to use
SELECTED_EXPERIMENT: str = "STANDARD"

