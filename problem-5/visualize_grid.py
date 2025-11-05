# visualize_grid.py

import gymnasium as gym
import argparse
import config

def visualize_grid(experiment_name: str) -> None:
    """
    Loads the environment for a given experiment and prints its
    rendered map to the console.
    """
    if experiment_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{experiment_name}' not found in config.py.")
        return

    exp_config = config.EXPERIMENTS[experiment_name]
    print(f"--- Visualizing Grid for: {exp_config['name']} ---")
    
    try:
        # Create the environment with the 'ansi' render mode for text output
        env = gym.make(
            exp_config["env_id"],
            map_name=exp_config["map_name"],
            is_slippery=exp_config["is_slippery"],
            render_mode='ansi'
        )
        # Resetting with the seed ensures we are looking at the exact instance
        env.reset(seed=exp_config["seed"])

        # The 'ansi' mode returns a string representation of the environment
        rendered_map = env.render()
        
        print("\nEnvironment Map:")
        print(rendered_map)
        
        env.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the grid for a FrozenLake experiment.")
    parser.add_argument("experiment_name", type=str, help="The name of the experiment to visualize (e.g., BASELINE_4x4).")
    args = parser.parse_args()
    
    visualize_grid(args.experiment_name)