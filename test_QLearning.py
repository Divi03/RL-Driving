import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# 1. Create the FrozenLake environment with render_mode specified
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode='human')  # 'human' will display the environment

# 2. Wrap the environment to support vectorized environments (this is recommended in Stable-Baselines3)
env = DummyVecEnv([lambda: env])

# 3. Initialize the DQN model
# model = DQN('MlpPolicy', env, verbose=1)

# # 4. Train the model
# model.learn(total_timesteps=10000)

# # 5. Save the model after training
# model.save("dqn_frozenlake_model")

# 6. Now let's load the saved model
loaded_model = DQN.load("/Users/divyansh/Downloads/dqn_frozenlake_model.zip")

# 7. Test the trained model (running the model after loading it)
obs = env.reset()
for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()  # This will render the FrozenLake grid
    if dones:
        print("Goal Reached!" if rewards == 1 else "Fallen into a hole!")
        # break

# 8. Close the environment after testing
env.close()
