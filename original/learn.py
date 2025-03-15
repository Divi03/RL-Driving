import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack
import pygame  # For manually terminating pygame window

# Create the environment
env = gym.make("CarRacing-v2", render_mode='human')

# Apply the grayscale wrapper (if you want grayscale observations)
env = GrayScaleObservation(env, keep_dim=True)

# Wrap the environment in a DummyVecEnv for parallel processing
env = DummyVecEnv([lambda: env])

# Stack frames for temporal context (using VecFrameStack)
env = VecFrameStack(env, 4, channels_order='last')

# Load the trained PPO model (make sure to change this to your model path)
model = PPO.load('/Users/divyansh/Downloads/best_model_1250000 (2).zip')

# Set the model to evaluation mode (optional, to prevent training updates)
model.set_env(env)

done = True
score = 0

try:
    while True:
        if done:
            # Reset the environment to start a new episode
            obs = env.reset()  # Unpack the observation and info
            score = 0  # Reset score at the start of each new episode

        # Predict the action using the PPO model
        action, _ = model.predict(obs)  # Model returns a tuple (action, states)
        print(action)
        # Take a step in the environment using the action
        obs, reward, done, info = env.step(action)

        # Update the score
        score += reward

        # Render the environment (uncomment if you want to visualize the game)
        env.render()

        # Print the score at the end of each episode
        if done:
            print(f"Episode ended. Total score: {score}")
            score = 0  # Reset score after each episode

except KeyboardInterrupt:
    print("Game interrupted. Closing environment.")

finally:
    # Close the environment after the loop ends to prevent rendering issues
    env.close()
    pygame.display.quit()  # Forcefully close the pygame display
    pygame.quit()  # Properly quit pygame
    print("Environment and pygame window closed.")
