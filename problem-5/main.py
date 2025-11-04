import gymnasium as gym
import numpy as np
import random
import time

# Import our configurations
import config

def main():
    # --- Step 0: Load the Selected Experiment Configuration ---
    exp_config = config.EXPERIMENTS[config.SELECTED_EXPERIMENT]
    print(f"--- Running Experiment: {exp_config['name']} ---")

    # --- Step 1: Setup the Environment ---
    seed = exp_config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make(
        exp_config["env_id"],
        map_name=exp_config["map_name"],
        is_slippery=exp_config["is_slippery"]
    )
    env.reset(seed=seed)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"Map: {exp_config['map_name']}, States: {num_states}, Actions: {num_actions}")

    # --- Step 2: Initialize Q-Table and Hyperparameters ---
    q_table = np.zeros((num_states, num_actions))

    # Load hyperparameters from the config file
    num_episodes = exp_config["num_episodes"]
    learning_rate = exp_config["learning_rate"]
    discount_factor = exp_config["discount_factor"]
    epsilon = exp_config["epsilon"]
    epsilon_decay_rate = exp_config["epsilon_decay_rate"]
    min_epsilon = exp_config["min_epsilon"]

    print("Hyperparameters loaded successfully.")

    # set the list of rewards
    rewards_per_episode = []

    print("Q-Table is ready. Starting training...")
    time.sleep(2) 

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # choose action
            if np.random.random() < epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # choose best action from q-table
                action = np.argmax(q_table[state, :])
            
            # take action and observe reward
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # update q-table
            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state, :])

            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            q_table[state, action] = new_value

            # update state and reward
            state = new_state
            episode_reward += reward

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        
        rewards_per_episode.append(episode_reward)

        if (episode + 1) % (num_episodes / 10) == 0:
            print(f"Episode {episode + 1} of {num_episodes} complete.")
    
    print("Training complete for {exp_config['name']}")

                

                
    
    env.close()

if __name__ == "__main__":
    main()