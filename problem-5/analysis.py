# analysis.py

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import argparse  # We use this to read arguments from the command line

# We need the config file to recreate the environment for evaluation
import config
import baselines

def load_results(experiment_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    q_table_path = os.path.join("data", f"{experiment_name}_q_table.npy")
    rewards_path = os.path.join("data", f"{experiment_name}_rewards.npy")

    if not os.path.exists(q_table_path) or not os.path.exists(rewards_path):
        print(f"Error: Data files not found for experiment '{experiment_name}'.")
        print("Please run the training script first.")
        return None, None
        
    q_table = np.load(q_table_path)
    rewards = np.load(rewards_path)
    return q_table, rewards

def calculate_success_rate(env: gym.Env, q_table: np.ndarray, num_episodes: int = 1000) -> float:
    successes: int = 0
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        while not done:
            action = np.argmax(q_table[state, :]) # Exploit only
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward == 1.0:
                successes += 1
                
    return (successes / num_episodes) * 100

def plot_learning_curve(
    rewards: np.ndarray, 
    experiment_name: str,
    random_rate: float,
    heuristic_rate: float
) -> None:
    plt.figure(figsize=(12, 8))
    
    window_size = 100
    moving_avg = np.convolve(
        rewards,
        np.ones(window_size)/window_size,
        mode='valid'
    )

    moving_avg_percent = moving_avg * 100
    plt.plot(
        range(window_size - 1, len(rewards)), 
        moving_avg_percent, 
        label=f'{window_size}-Episode Moving Average', 
        color='orange'
    )

    # draw baseline success rates
    plt.axhline(
        y=random_rate,
        color='r',
        linestyle='--',
        label=f'Random Policy ({random_rate:.2f}%)'
    )
    plt.axhline(
        y=heuristic_rate,
        color='g',
        linestyle=':',
        label=f'Heuristic Policy ({heuristic_rate:.2f}%)'
    )
    
    plt.title(f'Learning Curve for: {experiment_name} vs. Baseline Policies')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate % - Moving Average')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(-5, 100)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plot_path = os.path.join("plots", f"{experiment_name}_learning_curve.png")
    plt.savefig(plot_path)
    print(f"Learning curve plot saved to {plot_path}")
    plt.close() # Close the plot to free up memory

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the results of a Q-learning experiment.")
    parser.add_argument(
        "experiment_name", 
        type=str, 
        help="The name of the experiment to analyze (e.g., BASELINE_4x4)."
    )
    args = parser.parse_args()
    exp_name = args.experiment_name
    
    # Check if the experiment exists in the config
    if exp_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{exp_name}' not found in config.py.")
        return
        
    print(f"--- Analyzing Experiment: {exp_name} ---")
    
    # --- Load Data ---
    q_table, rewards = load_results(exp_name)
    if q_table is None or rewards is None:
        return 

    # assert rewards is not none
    assert rewards is not None, "Error: Rewards should not be None after initial check."

    # --- Recreate Environment for Evaluation ---
    exp_config = config.EXPERIMENTS[exp_name]
    env = gym.make(
        exp_config["env_id"],
        map_name=exp_config["map_name"],
        is_slippery=exp_config["is_slippery"]
    )

    # using a different seed from training for evaluation
    env.reset(seed=exp_config["seed"] + 1) 

    # calculate the success rates for all 
    print("\nCalculating success rates...")
    q_learning_success_rate = calculate_success_rate(env, q_table)
    random_success_rate = baselines.run_random_policy(env)
    heuristic_success_rate = baselines.run_heuristic_policy(env, exp_config["map_name"])

    print(f"\n ->   Q-Learning Success Rate: {q_learning_success_rate:.2f}%")
    print(f" ->   Random Success Rate: {random_success_rate:.2f}%")
    print(f" ->   Heuristic Success Rate: {heuristic_success_rate:.2f}%")

    # generate full plot 
    plot_learning_curve(
        rewards,
        exp_name,
        random_success_rate,
        heuristic_success_rate
    )

    env.close()

if __name__ == "__main__":
    main()