# import numpy as np
# import gym
# import matplotlib.pyplot as plt

# # Function to convert RGB to grayscale
# def rgb2gray(rgb, norm=True):
#     gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # RGB to grayscale
#     if norm:
#         gray = gray / 128.0 - 1.0  # Normalize to [-1, 1]
#     return gray

# # Preprocessing function to resize and convert to grayscale
# def preprocess(frame, resize_shape=(96, 96)):
#     gray_frame = rgb2gray(frame)
#     # Resize the image to (96, 96)
#     gray_frame_resized = np.resize(gray_frame, resize_shape)
#     return gray_frame_resized

# # Load 5 environments
# envs = [gym.make('CarRacing-v2', render_mode='rgb_array') for _ in range(5)]

# # Initialize the figure for plotting (larger figure size)
# plt.figure(figsize=(3, 9))

# for i, env in enumerate(envs):
#     # Reset the environment
#     state, _ = env.reset()

#     # Step the environment (taking a random action as an example)
#     action = np.random.uniform(-1, 1, size=env.action_space.shape[0])  # Random action
#     frame, _, _, _, _ = env.step(action)
    
#     # Preprocess the frame (convert to grayscale)
#     preprocessed_frame = preprocess(frame)
    
#     # Plot the colored image (original)
#     plt.subplot(5, 2, 2 * i + 1)
#     plt.imshow(frame)
#     plt.axis('off')  # Remove axis

#     # Plot the grayscale image (preprocessed)
#     plt.subplot(5, 2, 2 * i + 2)
#     plt.imshow(preprocessed_frame, cmap='gray')
#     plt.axis('off')  # Remove axis

# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.4, wspace=0.2)

# # Show the plots
# plt.show()


# import numpy as np
# import gym
# import matplotlib.pyplot as plt
# import time

# # Function to convert RGB to grayscale
# def rgb2gray(rgb, norm=True):
#     gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # RGB to grayscale
#     if norm:
#         gray = gray / 128.0 - 1.0  # Normalize to [-1, 1]
#     return gray

# # Preprocessing function to resize and convert to grayscale
# def preprocess(frame, resize_shape=(96, 96)):
#     gray_frame = rgb2gray(frame)
#     # Resize the image to (96, 96)
#     gray_frame_resized = np.resize(gray_frame, resize_shape)
#     return gray_frame_resized

# # Load 5 environments
# envs = [gym.make('CarRacing-v2', render_mode='rgb_array') for _ in range(5)]

# # Initialize the figure for plotting (larger figure size)
# plt.figure(figsize=(4, 9))  # Increased width for an extra column

# for i, env in enumerate(envs):
#     # Reset the environment
#     state, _ = env.reset()

#     # Capture the initial frame (first screenshot)
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action

    
#     # Preprocess the initial frame (convert to grayscale)
#     preprocessed_frame = preprocess(frame)
    
#     # Plot the original colored image (initial)
#     plt.subplot(5, 3, 3 * i + 1)
#     plt.imshow(frame)
#     plt.axis('off')  # Remove axis

#     # Plot the grayscale image (initial)
#     plt.subplot(5, 3, 3 * i + 2)
#     plt.imshow(preprocessed_frame, cmap='gray')
#     plt.axis('off')  # Remove axis

#     # Wait for 2 seconds (simulating time passing in the environment)
#     # time.sleep(2)

#     # Capture the second frame after 2 seconds

#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
#     frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
    
#     frame_after_2_sec, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action
    
#     # Preprocess the second frame (convert to grayscale)
#     preprocessed_frame_after_2_sec = preprocess(frame_after_2_sec)
    
#     # Plot the second screenshot (colored image after 2 seconds)
#     plt.subplot(5, 3, 3 * i + 3)
#     plt.imshow(frame_after_2_sec)
#     plt.axis('off')  # Remove axis

# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.4, wspace=0.2)

# # Show the plots
# plt.show()



import numpy as np
import gym
import matplotlib.pyplot as plt

# Function to convert RGB to grayscale
def rgb2gray(rgb, norm=True):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # RGB to grayscale
    if norm:
        gray = gray / 128.0 - 1.0  # Normalize to [-1, 1]
    return gray

# Preprocessing function to resize and convert to grayscale
def preprocess(frame, resize_shape=(96, 96)):
    gray_frame = rgb2gray(frame)
    # Resize the image to (96, 96)
    gray_frame_resized = np.resize(gray_frame, resize_shape)
    return gray_frame_resized

# Load 5 environments
envs = [gym.make('CarRacing-v2', render_mode='rgb_array') for _ in range(5)]

# Initialize the figure for plotting (larger figure size)
plt.figure(figsize=(5, 9))  # Increased width for an extra column

for i, env in enumerate(envs):
    # Reset the environment
    state, _ = env.reset()

    # Capture the initial frame (first screenshot)
    frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action

    # Preprocess the initial frame (convert to grayscale)
    preprocessed_frame = preprocess(frame)
    
    # Plot the original colored image (initial)
    plt.subplot(5, 4, 4 * i + 1)
    plt.imshow(frame)
    plt.axis('off')  # Remove axis

    # Plot the grayscale image (initial)
    plt.subplot(5, 4, 4 * i + 2)
    plt.imshow(preprocessed_frame, cmap='gray')
    plt.axis('off')  # Remove axis

    # Perform multiple actions to simulate waiting for a few seconds
    for _ in range(20):  # 20 random actions (arbitrary, simulates ~2 seconds)
        frame, _, _, _, _ = env.step(np.random.uniform(-1, 1, size=env.action_space.shape[0]))  # Random action

    # Capture the second frame after a few actions (about 2 seconds later)
    frame_after_2_sec = frame  # This is the last frame after actions

    # Preprocess the second frame (convert to grayscale)
    preprocessed_frame_after_2_sec = preprocess(frame_after_2_sec)
    
    # Plot the second screenshot (colored image after 2 seconds)
    plt.subplot(5, 4, 4 * i + 3)
    plt.imshow(frame_after_2_sec)
    plt.axis('off')  # Remove axis

    # Plot the grayscale version of the second screenshot
    plt.subplot(5, 4, 4 * i + 4)
    plt.imshow(preprocessed_frame_after_2_sec, cmap='gray')
    plt.axis('off')  # Remove axis

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.2)

# Show the plots
plt.show()
