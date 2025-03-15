# # Resize 48x48
# import gymnasium as gym
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# class ResizeWrapper(gym.ObservationWrapper):
#     def __init__(self, env, new_size=(48, 48)):
#         super().__init__(env)
#         self.new_size = new_size
#         # Update observation space to match the new size
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(new_size[0], new_size[1], 3), dtype=np.uint8)

#     def observation(self, obs):
#         # Resize the image observation
#         return np.array(Image.fromarray(obs).resize(self.new_size))

# def rgb_to_grayscale(rgb_img):
#     # Convert the RGB image to grayscale by averaging the color channels
#     return np.dot(rgb_img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# # Create your Gym environment (example: CarRacing-v2)
# env = gym.make("CarRacing-v2", render_mode='rgb_array')

# # Apply the resize wrapper to preprocess frames to 48x48
# env = ResizeWrapper(env)

# # Initialize variables
# done = False
# frames = []

# # Reset the environment and get the initial observation
# obs, info = env.reset()

# # Collect frames until step 60
# for step in range(60):  # Collect frames until step 60
#     action = env.action_space.sample()  # Take a random action
#     obs, reward, done, truncated, info = env.step(action)

#     if done or truncated:
#         break

# # Capture the frames at step 60
# original_frame = obs  # Original frame at step 60

# # Convert original frame to grayscale
# grayscale_frame = rgb_to_grayscale(original_frame)

# # Resize both the original frame and the grayscale frame to 48x48
# original_resized_frame = np.array(Image.fromarray(original_frame).resize((48, 48)))
# grayscale_resized_frame = np.array(Image.fromarray(grayscale_frame).resize((48, 48)))

# # Visualize the collected frames using Matplotlib
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# # Display Original Frame (96x96) and Resized Frame (48x48)
# axes[0, 0].imshow(original_frame)
# axes[0, 0].axis('off')
# axes[0, 0].set_title("Original Frame (96x96)")

# axes[0, 1].imshow(original_resized_frame)
# axes[0, 1].axis('off')
# axes[0, 1].set_title("Resized Frame (48x48)")

# # Display Grayscale Frame (96x96) and Resized Grayscale Frame (48x48)
# axes[1, 0].imshow(grayscale_frame, cmap='gray')
# axes[1, 0].axis('off')
# axes[1, 0].set_title("Grayscale Frame (96x96)")

# axes[1, 1].imshow(grayscale_resized_frame, cmap='gray')
# axes[1, 1].axis('off')
# axes[1, 1].set_title("Resized Grayscale Frame (48x48)")

# # Adjust layout to prevent overlapping titles/labels
# plt.tight_layout()
# plt.show()

# # Close the environment after visualization
# env.close()



import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create your Gym environment (example: CarRacing-v2)
env = gym.make("CarRacing-v2", render_mode='human')

# Initialize variables
done = False
frames = []

# Reset the environment and get the initial observation
obs, info = env.reset()

# Collect frames between step 60 and step 70
for step in range(70):  # Collect frames until step 70
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, truncated, info = env.step(action)

    if step >= 60:  # Start collecting frames from step 60
        frames.append(obs)  # Store the frame

    if done or truncated:
        break

# Now we have collected frames from 60 to 70

# Visualize the collected frames using Matplotlib
fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # Create a 2x5 grid for 10 frames

# Loop through the frames and plot them
for i, frame in enumerate(frames):
    ax = axes[i // 5, i % 5]  # Get the correct subplot axis
    ax.imshow(frame)  # Plot the frame
    ax.axis('off')  # Turn off axis for cleaner visualization
    ax.set_title(f"Frame {i+60}")  # Set title with the frame number (60 to 69)

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()
plt.show()

# Close the environment after visualization
env.close()
