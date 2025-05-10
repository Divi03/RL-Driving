import gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import GrayScaleObservation, FrameStack
import pygame
from pygame.locals import *

# Initialize Pygame for keyboard input (for manual control)
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("CarRacing Control")

# Preprocessing functions
def apply_gaussian_blur(image, ksize=5):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def adjust_brightness(image, factor=1.5):
    """Adjust brightness of the image."""
    image = np.clip(np.float32(image) * factor, 0, 255)
    return np.uint8(image)

def adjust_saturation(image, factor=1.5):
    """Adjust saturation of the image by converting to HSV and scaling the S channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def extract_multi_center_color(image, region_size=7, offset_y=10):
    """Extract color from multiple regions around the center of the image."""
    h, w, _ = image.shape
    cx = w // 2
    cy = h // 2
    offsets = [-offset_y, 0, offset_y]
    colors = []
    for off in offsets:
        y1 = cy + off - region_size // 2
        y2 = cy + off + region_size // 2
        x1 = cx - region_size // 2
        x2 = cx + region_size // 2
        region = image[y1:y2, x1:x2]
        colors.append(np.mean(region, axis=(0, 1)))
    avg_color = np.mean(colors, axis=0)
    return avg_color

def extract_track_by_center_color_rgb(image, center_color, tolerance=25):
    """Create a mask of the track based on the center color."""
    diff = np.abs(image.astype(np.int16) - center_color.astype(np.int16))
    diff_sum = np.sum(diff, axis=2)
    mask = np.uint8(diff_sum < tolerance) * 255

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

# Preprocessing wrapper to apply the same logic in the model environment
class TrackPreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.center_color = None
        self.frame_count = 0

    def observation(self, obs):
        self.frame_count += 1

        # Apply image preprocessing steps
        obs = apply_gaussian_blur(obs, ksize=5)
        obs = adjust_brightness(obs, factor=0.9)
        obs = adjust_saturation(obs, factor=1.5)

        # Extract center color once after 15 frames
        if self.frame_count == 15:
            self.center_color = extract_multi_center_color(obs, region_size=7)

        # Apply track masking if center color is available
        if self.center_color is not None:
            track_mask = extract_track_by_center_color_rgb(obs, self.center_color, tolerance=20)
            obs = cv2.cvtColor(track_mask, cv2.COLOR_GRAY2RGB)

        return obs

# Create the environment
env = gym.make("CarRacing-v2", domain_randomize=True)

# Apply the preprocessing wrapper to environment
env = TrackPreprocessingWrapper(env)

# Wrap the environment for grayscale observation (necessary for PPO with image-based policies)
env = GrayScaleObservation(env, keep_dim=True)

# Add FrameStack wrapper (stack 4 frames)
env = FrameStack(env, 4)

# Create a vectorized environment (necessary for stable-baselines3)
env = DummyVecEnv([lambda: env])

# Load pre-trained model
model_path = '/Users/divyansh/Downloads/best_model_1000000 (1).zip'  # Replace with actual path to your model
model = PPO.load(model_path)
print("Loaded pre-trained model.")

# To visualize, run a loop to display the trained model's performance (optional)
obs = env.reset()  # Make sure to reset the environment before starting the loop
done = False

# Run inference loop (the trained model will control the car)
while not done:
    action, _states = model.predict(obs, deterministic=True)  # Predict action based on the current observation
    obs, reward, done, truncated, info = env.step(action)  # Take the action and get next observation

env.close()  # Close the environment after the loop ends




