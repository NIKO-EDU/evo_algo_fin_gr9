# config.py

# A dictionary to hold all our experiment configurations
EXPERIMENTS = {
    "BASELINE_4x4": {
        "name": "Baseline 4x4 - Far-Sighted",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 20000,
        "learning_rate": 0.1,         # Alpha: How quickly the agent learns
        "discount_factor": 0.99,      # Gamma: Importance of future rewards
        "epsilon": 1.0,               # Initial exploration rate
        "epsilon_decay_rate": 0.0001, # Rate of exploration decay
        "min_epsilon": 0.01,          # Minimum exploration rate
        "seed": 42                    # For reproducibility
    },
    "HARDER_8x8": {
        "name": "Harder 8x8 - Increased Complexity",
        "env_id": "FrozenLake-v1",
        "map_name": "8x8",
        "is_slippery": True,
        "num_episodes": 100000,       # Needs many more episodes to learn
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.0001, # Slower decay might be needed, but start here
        "min_epsilon": 0.01,
        "seed": 42
    },
    "MYOPIC_AGENT_4x4": {
        "name": "Myopic 4x4 - Short-Sighted Agent",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 20000,
        "learning_rate": 0.1,
        "discount_factor": 0.90,      # Lower gamma, cares less about the future
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.0001,
        "min_epsilon": 0.01,
        "seed": 42
    },
    "CAUTIOUS_AGENT_4x4": {
        "name": "Cautious 4x4 - Slower Learning",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 20000,
        "learning_rate": 0.01,        # Lower alpha, learns more slowly
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.0001,
        "min_epsilon": 0.01,
        "seed": 42
    }
}
SELECTED_EXPERIMENT = "BASELINE_4x4"