# import gymnasium as gym
# import numpy as np
# import pickle
# import pygame
# import os

# def collect_user_data(env_name="CarRacing-v2", num_episodes=5, save_dir="episode_data", video_dir="videos"):
#     # Create the environment
#     env = gym.make(env_name, render_mode="rgb_array")  # Use rgb_array for frame capture
#     metadata = env.metadata

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)

#     # Initialize Pygame for manual control
#     pygame.init()
#     screen = pygame.display.set_mode((600, 400))  # Optional window for control

#     for episode in range(num_episodes):
#         print(f"Starting Episode {episode + 1}/{num_episodes}")
        
#         # Prepare to save video frames
#         frames = []
        
#         state, info = env.reset()  # Get the initial state and info
#         done = False
#         data = []  # To store state-action-reward-done tuples for this episode

#         while not done:
#             # Get the frame from the environment
#             frame = env.render()  # Get rendered frame for video
#             frames.append(frame)

#             # Convert frame to pygame surface
#             frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
#             screen.blit(frame_surface, (0, 0))
#             pygame.display.flip()  # Update the display
            
#             action = np.array([0.0, 0.0, 0.0])  # Initialize action (steering, gas, brake)

#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     env.close()
#                     pygame.quit()
#                     return  # Exit if the window is closed

#             # Handle key events for manual control
#             keys = pygame.key.get_pressed()
#             action[0] = float(keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])  # Steering: Right(1), Left(-1)
#             action[1] = float(keys[pygame.K_UP])  # Gas
#             action[2] = float(keys[pygame.K_DOWN])  # Brake

#             # Step in the environment using the action
#             next_state, reward, terminated, truncated, info = env.step(action)

#             # Store the collected data as a tuple: (state, action, reward, next_state, done)
#             data.append((state, action, reward, next_state, terminated or truncated))

#             # Update the state for the next step
#             state = next_state
#             done = terminated or truncated

#         print(f"Episode {episode + 1} completed.")
        
#         # Save the episode data
#         with open(os.path.join(save_dir, f'user_data_episode_{episode + 1}.pkl'), 'wb') as f:
#             pickle.dump(data, f)
        
#         # Save video frames for the episode
#         save_video(frames, os.path.join(video_dir, f'episode_{episode + 1}.mp4'), metadata)

#     env.close()


# def save_video(frames, video_path, metadata, fps=30):
#     """
#     Save the recorded frames as a video using OpenCV or imageio
#     """
#     import imageio

#     imageio.mimsave(video_path, frames, fps=fps)
#     print(f"Video saved: {video_path}")

# # Run the script to record 5 episodes
# if __name__ == "__main__":
#     collect_user_data(env_name="CarRacing-v2", num_episodes=5)

import gymnasium as gym
import numpy as np
import pickle
import os
from gym.wrappers import RecordVideo

def simulate_and_record(env_name="CarRacing-v2", data_file="user_data_episode_1.pkl", video_dir="videos"):
    # Create the environment with discrete actions
    env = gym.make(env_name, render_mode="rgb_array")  # Use rgb_array for frame capture
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)  # Record every episode

    # Load the recorded data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # Simulate using the loaded data
    for episode_index, (state, action, reward, next_state, done) in enumerate(data):
        print(f"Simulating Episode {episode_index + 1}")

        # Reset the environment at the start of the episode
        obs, info = env.reset()
        done = False
        episode_reward = 0  # To keep track of total reward in the episode

        while not done:
            # Render the environment and get the current frame
            frame = env.render()  # Get rendered frame for video
            if frame is not None:
                # Optionally do something with the frame if needed
                pass
            
            # Step through the environment using recorded action
            action_taken = np.array([action])  # Convert action to a numpy array
            next_state, reward, terminated, truncated, info = env.step(action_taken)

            # Accumulate the reward
            episode_reward += reward
            
            # Check for episode termination
            done = terminated or truncated

            # Update state for next step (not used here since we're following the recorded actions)
            state = next_state

        print(f"Episode {episode_index + 1} completed with total reward: {episode_reward}")

    env.close()

# Run the simulation and recording
if __name__ == "__main__":
    simulate_and_record(env_name="CarRacing-v2", data_file="user_data_episode_1.pkl", video_dir="videos")
