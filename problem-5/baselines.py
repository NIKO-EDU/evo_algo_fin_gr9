# baselines.py

import gymnasium as gym
import numpy as np

def run_random_policy(env: gym.Env, num_episodes: int = 1000) -> float:
    
    successes = 0
    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample() # Always choose a random action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward == 1.0:
                successes += 1
    return (successes / num_episodes) * 100

def run_heuristic_policy(env: gym.Env, map_name: str, num_episodes: int = 1000) -> float:
    successes = 0
    map_size = int(np.sqrt(env.observation_space.n))
    goal_state = map_size * map_size - 1

    # Action mapping: 0=Left, 1=Down, 2=Right, 3=Up
    DOWN, RIGHT = 1, 2

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        while not done:
            current_row, current_col = divmod(state, map_size)
            goal_row, goal_col = divmod(goal_state, map_size)
            
            # Simple heuristic: prefer Down or Right if not at goal row/col
            if current_row < goal_row:
                action = DOWN
            elif current_col < goal_col:
                action = RIGHT
            else:
                # If we reach here, we're at or past the goal position
                # (current_row >= goal_row AND current_col >= goal_col)
                # In this case, prefer RIGHT then DOWN (though we should be at goal)
                action = RIGHT

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward == 1.0:
                successes += 1
    return (successes / num_episodes) * 100