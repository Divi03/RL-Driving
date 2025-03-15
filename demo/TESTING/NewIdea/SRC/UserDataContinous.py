import gymnasium as gym
import numpy as np
import pickle
import pygame

def collect_user_data(env_name="CarRacing-v2", num_episodes=1):
    env = gym.make(env_name, render_mode="human", continuous=True)
    data = []

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # Adjust size as necessary

    for episode in range(num_episodes):
        state, info = env.reset()  # Reset returns the initial state and info
        done = False

        while not done:
            env.render()  # Render the environment
            action = np.array([0.0, 0.0, 0.0])  # Initialize action

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return data  # Exit if the window is closed

            # Check for key presses
            keys = pygame.key.get_pressed()
            action[0] = float(keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])  # Steering
            action[1] = float(keys[pygame.K_UP])  # Gas
            action[2] = float(keys[pygame.K_DOWN])  # Brake

            # Take a step in the environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store the collected data as a tuple
            # Format: (state, action, reward, next_state, done)
            data.append((state, action, reward, next_state, terminated or truncated))

            state = next_state
            done = terminated or truncated  # Update done condition

        print("\nEpisode completed.")

    env.close()

    # Save the collected data for RL model training
    with open('user_data4.pkl', 'wb') as f:
        pickle.dump(data, f)

# Run data collection
if __name__ == "__main__":
    collect_user_data()
