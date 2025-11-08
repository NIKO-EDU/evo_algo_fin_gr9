# config.py

# A dictionary to hold all our experiment configurations
EXPERIMENTS = {
    "4x4_BASIC": {
        "name": "4x4 Basic",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 20000,
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.000066, # Decays over ~15k episodes
        "min_epsilon": 0.01,
        "seed": 42
    },
    "8x8_HARD": {
        "name": "8x8 Hard",
        "env_id": "FrozenLake-v1",
        "map_name": "8x8",
        "is_slippery": True,
        "num_episodes": 60000,
        "learning_rate": 0.1,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.0000222, # Decays over ~45k episodes
        "min_epsilon": 0.01,
        "seed": 42
    },
    "4x4_MYOPIC": {
        "name": "4x4 Myopic (Low Gamma)",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 20000,
        "learning_rate": 0.1,
        "discount_factor": 0.90,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.000066, # Decays over ~15k episodes
        "min_epsilon": 0.01,
        "seed": 42
    },
    "4x4_CAUTIOUS": {
        "name": "4x4 Cautious (Low Alpha)",
        "env_id": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "num_episodes": 40000,
        "learning_rate": 0.01,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_decay_rate": 0.000033, # Decays over ~30k episodes
        "min_epsilon": 0.01,
        "seed": 42
    }
}