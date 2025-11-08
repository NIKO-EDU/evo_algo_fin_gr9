# analysis.py

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import argparse

# We need the config file to recreate the environment for evaluation
import config
import baselines

def load_results(experiment_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Loads Q-table, rewards, and epsilon history."""
    q_table_path = os.path.join("data", f"{experiment_name}_q_table.npy")
    rewards_path = os.path.join("data", f"{experiment_name}_rewards.npy")
    epsilon_path = os.path.join("data", f"{experiment_name}_epsilon.npy")

    if not all(os.path.exists(p) for p in [q_table_path, rewards_path, epsilon_path]):
        print(f"Error: Data files not found for experiment '{experiment_name}'. Please run training script first.")
        return None, None, None
        
    q_table = np.load(q_table_path)
    rewards = np.load(rewards_path)
    epsilon_history = np.load(epsilon_path)
    return q_table, rewards, epsilon_history

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
    epsilon_history: np.ndarray,
    experiment_name: str,
    random_rate: float,
    heuristic_rate: float,
    peak_info: dict
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # --- Plot 1: Success Rate (left y-axis) ---
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate % - Moving Average', color='orange')
    ax1.set_ylim(-5, 100)
    ax1.grid(True)

    # Calculate and plot 100-episode moving average
    window_size_100 = 100
    moving_avg_100 = np.convolve(rewards, np.ones(window_size_100)/window_size_100, mode='valid')
    moving_avg_percent_100 = moving_avg_100 * 100
    episodes_100 = range(window_size_100 - 1, len(rewards))
    ax1.plot(episodes_100, moving_avg_percent_100, label=f'{window_size_100}-Episode Moving Average', color='orange')
    
    # Calculate and plot 500-episode moving average
    window_size_500 = 500
    # Only plot if there is enough data for the window
    if len(rewards) > window_size_500:
        moving_avg_500 = np.convolve(rewards, np.ones(window_size_500)/window_size_500, mode='valid')
        moving_avg_percent_500 = moving_avg_500 * 100
        episodes_500 = range(window_size_500 - 1, len(rewards))
        ax1.plot(episodes_500, moving_avg_percent_500, label=f'{window_size_500}-Episode Moving Average', color='c', linewidth=2)

    ax1.tick_params(axis='y', labelcolor='orange')

    # Plot baselines
    ax1.axhline(y=random_rate, color='r', linestyle='--', label=f'Random Policy ({random_rate:.2f}%)')
    ax1.axhline(y=heuristic_rate, color='g', linestyle=':', label=f'Heuristic Policy ({heuristic_rate:.2f}%)')

    # Plot a vertical line at the peak performance (based on 100-ep window)
    peak_episode = peak_info['episode']
    peak_value = peak_info['value']
    ax1.axvline(x=peak_episode, color='b', linestyle='-.', 
                label=f'Peak Performance ({peak_value:.2f}% at Ep {peak_episode})')

    # --- Plot 2: Epsilon Decay (right y-axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon (Exploration Rate)', color='purple')
    ax2.plot(epsilon_history, color='purple', linestyle=':', alpha=0.7, label='Epsilon Value')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor='purple')

    # --- Final Touches ---
    fig.suptitle(f'Learning Curve for: {experiment_name} vs. Baselines')
    
    # Combine legends from both axes and place in the upper left
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plot_path = os.path.join("plots", f"{experiment_name}_learning_curve.png")
    plt.savefig(plot_path, dpi=300) # Save with high resolution
    print(f"Learning curve plot saved to {plot_path}")
    plt.close()

def main() -> None:
    # --- Step 0: Load Configuration ---
    parser = argparse.ArgumentParser(description="Analyze the results of a Q-learning experiment.")
    parser.add_argument(
        "experiment_name", 
        type=str, 
        help="The name of the experiment to analyze (e.g., 4x4_BASIC)."
    )
    args = parser.parse_args()
    exp_name = args.experiment_name
    
    if exp_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{exp_name}' not found in config.py.")
        return
        
    print(f"--- Analyzing Experiment: {exp_name} ---")
    
    # --- Load Data ---
    q_table, rewards, epsilon_history = load_results(exp_name)
    if q_table is None:
        return 
    assert rewards is not None and epsilon_history is not None

    # --- Recreate Environment for Evaluation ---
    exp_config = config.EXPERIMENTS[exp_name]
    env = gym.make(
        exp_config["env_id"],
        map_name=exp_config["map_name"],
        is_slippery=exp_config["is_slippery"]
    )
    env.reset(seed=exp_config["seed"] + 1)

    # --- Calculate Success Rates ---
    print("\nCalculating success rates...")
    q_learning_success_rate = calculate_success_rate(env, q_table)
    random_success_rate = baselines.run_random_policy(env)
    heuristic_success_rate = baselines.run_heuristic_policy(env, exp_config["map_name"])

    print(f"\n ->   Q-Learning Success Rate: {q_learning_success_rate:.2f}%")
    print(f" ->   Random Success Rate: {random_success_rate:.2f}%")
    print(f" ->   Heuristic Success Rate: {heuristic_success_rate:.2f}%")

    # --- Find Peak Performance (based on 100-episode window for sensitivity) ---
    window_size = 100
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    peak_value_fraction = np.max(moving_avg)
    peak_episode_index = np.argmax(moving_avg) + (window_size - 1)
    
    peak_info = {
        "value": peak_value_fraction * 100,
        "episode": peak_episode_index
    }
    print(f" ->   Peak Moving Average (100ep): {peak_info['value']:.2f}% at episode ~{peak_info['episode']}")

    # --- Generate Full Plot ---
    plot_learning_curve(
        rewards,
        epsilon_history,
        exp_name,
        random_success_rate,
        heuristic_success_rate,
        peak_info
    )

    env.close()

if __name__ == "__main__":
    main()