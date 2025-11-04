import gymnasium as gym
import numpy as np
import random
import time

# Import our configurations
import config

def train_agent(env, q_table, num_episodes, learning_rate, discount_factor, 
                epsilon, min_epsilon, epsilon_decay_rate):
    print("Training started...")
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

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
            episode_reward += reward

        # Decay epsilon after the episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        rewards_per_episode.append(episode_reward)

        # Print progress
        if (episode + 1) % (num_episodes / 10) == 0:
            print(f"  -> Episode {episode + 1}/{num_episodes} complete.")
            
    print("Training finished.")
    return q_table, rewards_per_episode


def main():
    # --- Step 0: Load Configuration ---
    exp_config = config.EXPERIMENTS[config.SELECTED_EXPERIMENT]
    print(f"--- Running Experiment: {exp_config['name']} ---")

    # --- Step 1: Setup Environment ---
    seed = exp_config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(
        exp_config["env_id"],
        map_name=exp_config["map_name"],
        is_slippery=exp_config["is_slippery"]
    )
    env.reset(seed=seed)

    # --- Step 2: Initialize Agent ---
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    print(f"Map: {exp_config['map_name']}, States: {num_states}, Actions: {num_actions}")
    
    # --- Step 3: Train the Agent ---
    trained_q_table, rewards = train_agent(
        env=env,
        q_table=q_table,
        num_episodes=exp_config["num_episodes"],
        learning_rate=exp_config["learning_rate"],
        discount_factor=exp_config["discount_factor"],
        epsilon=exp_config["epsilon"],
        min_epsilon=exp_config["min_epsilon"],
        epsilon_decay_rate=exp_config["epsilon_decay_rate"]
    )

    print(f"\n--- Training complete for {exp_config['name']} ---")
    print(f"Successfully received trained Q-table with shape {trained_q_table.shape}")
    print(f"Successfully received rewards for {len(rewards)} episodes.")
    
    env.close()

if __name__ == "__main__":
    main()