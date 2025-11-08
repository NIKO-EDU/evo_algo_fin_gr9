import gymnasium as gym
from typing import Any
import argparse
import numpy as np
import random
import time
import os

# Import our configurations
import config

def train_agent(
    env: gym.Env, 
    q_table: np.ndarray, 
    num_episodes: int, 
    learning_rate: float, 
    discount_factor: float, 
    epsilon: float, 
    min_epsilon: float, 
    epsilon_decay_rate: float
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Trains the agent using the Q-learning algorithm.
    Returns the trained Q-table, a list of rewards per episode,
    and a list of epsilon values per episode.
    """
    print("Training started...")
    rewards_per_episode: list[float] = []
    epsilon_history: list[float] = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done: bool = False
        episode_reward: float = 0.0

        while not done:
            # 1. Choose an Action (Epsilon-Greedy Strategy)
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit

            # 2. Take the Action
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Update the Q-Table (Bellman Equation)
            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state, :])
            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            q_table[state, action] = new_value

            # 4. Update state and total reward
            state = new_state
            episode_reward += float(reward)

        # After the episode is done, record history and decay epsilon
        epsilon_history.append(epsilon)
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        rewards_per_episode.append(episode_reward)

        # Print progress
        if (episode + 1) % (num_episodes / 10) == 0:
            print(f"  -> Episode {episode + 1}/{num_episodes} complete.")
            
    print("Training finished.")
    return q_table, rewards_per_episode, epsilon_history

def save_results(
    experiment_name: str,
    q_table: np.ndarray,
    rewards: list[float],
    epsilon_history: list[float]
) -> None:
    print("Saving results...")

    if not os.path.exists("data"): os.makedirs("data")

    q_table_path = os.path.join("data", f"{experiment_name}_q_table.npy")
    rewards_path = os.path.join("data", f"{experiment_name}_rewards.npy")
    epsilon_path = os.path.join("data", f"{experiment_name}_epsilon.npy")

    np.save(q_table_path, q_table)
    np.save(rewards_path, np.array(rewards))
    np.save(epsilon_path, np.array(epsilon_history))

    print(f"Results saved to data/{experiment_name}_*")

def main() -> None:
    # --- Step 0: Load Configuration FROM COMMAND LINE ---
    parser = argparse.ArgumentParser(description="Train a Q-learning agent for a given experiment.")
    parser.add_argument("experiment_name", type=str, help="The name of the experiment to run (e.g., 4x4_BASIC).")
    args = parser.parse_args()
    exp_name = args.experiment_name

    if exp_name not in config.EXPERIMENTS:
        print(f"Error: Experiment '{exp_name}' not found in config.py.")
        return

    exp_config: dict[str, Any]= config.EXPERIMENTS[exp_name]
    print(f"--- Running Experiment: {exp_config['name']} ---")

    # --- Step 1: Setup Environment ---
    seed: int = int(exp_config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    env: gym.Env = gym.make(
        str(exp_config["env_id"]),
        map_name=str(exp_config["map_name"]),
        is_slippery=bool(exp_config["is_slippery"])
    )
    env.reset(seed=seed)

    # --- Step 2: Initialize Agent ---
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_states: int = int(env.observation_space.n)
    num_actions: int = int(env.action_space.n)
    q_table: np.ndarray = np.zeros((num_states, num_actions))
    print(f"Map: {exp_config['map_name']}, States: {num_states}, Actions: {num_actions}")
    
    # --- Step 3: Train the Agent ---
    trained_q_table: np.ndarray
    rewards: list[float]
    epsilon_history: list[float]
    
    trained_q_table, rewards, epsilon_history = train_agent(
        env=                env,
        q_table=            q_table,
        num_episodes=       int(exp_config["num_episodes"]),
        learning_rate=      float(exp_config["learning_rate"]),
        discount_factor=    float(exp_config["discount_factor"]),
        epsilon=            float(exp_config["epsilon"]),
        min_epsilon=        float(exp_config["min_epsilon"]),
        epsilon_decay_rate= float(exp_config["epsilon_decay_rate"])
    )
    print(f"\n--- Training complete for {exp_config['name']} ---")

    # --- Step 4: Save Results ---
    save_results(
        exp_name,
        trained_q_table,
        rewards,
        epsilon_history
    )
    
    env.close()

if __name__ == "__main__":
    main()