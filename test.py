# # import pickle
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Load the .pkl file
# # file_path = '/Applications/Files/SEM_7/MAJOR/demo/GymCarCNNClassifier/models/model1_3_epochs_history.pkl'  # Adjust the file path
# # with open(file_path, 'rb') as file:
# #     history_data = pickle.load(file)

# # # # Inspect the type of the data
# # # print(f"Type of data in the file: {type(history_data)}")

# # # # If the data is a dictionary, let's print its keys and inspect their values
# # # if isinstance(history_data, dict):
# # #     # print(f"Keys in the dictionary: {history_data.keys()}")
    
# # #     # Inspect the first 5 items for each key if the value is a list
# # #     for key, value in history_data.items():
# # #         print(f"Key: {key}, Value type: {type(value)}")
# # #         if isinstance(value, (list, np.ndarray)):  # If the value is a list or ndarray
# # #             print(f"First 5 values for {key}: {value[:5]}")  # Show first 5 elements
# # #         else:
# # #             print(f"First 5 values for {key}: {value}")  # Show first value or just describe
# # # else:
# # #     print("The data is not a dictionary, it's of type:", type(history_data))
# # #     # If it's not a dictionary, inspect the first 5 items if it's a list or other type
# # #     print(f"First 5 elements: {history_data[:5] if isinstance(history_data, (list, np.ndarray)) else history_data}")


# # # Extract the data from the dictionary
# # loss = history_data['loss']
# # accuracy = history_data['accuracy']
# # val_loss = history_data['val_loss']
# # val_accuracy = history_data['val_accuracy']

# # # Plotting all the data

# # plt.figure(figsize=(12, 10))

# # # Plot Training Loss
# # plt.subplot(2, 2, 1)
# # plt.plot(loss, label='Training Loss', color='blue')
# # plt.title('Training Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()

# # # Plot Validation Loss
# # plt.subplot(2, 2, 2)
# # plt.plot(val_loss, label='Validation Loss', color='red')
# # plt.title('Validation Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()

# # # Plot Training Accuracy
# # plt.subplot(2, 2, 3)
# # plt.plot(accuracy, label='Training Accuracy', color='green')
# # plt.title('Training Accuracy')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()

# # # Plot Validation Accuracy
# # plt.subplot(2, 2, 4)
# # plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
# # plt.title('Validation Accuracy')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()

# # # Adjust layout for better spacing between plots
# # plt.tight_layout()

# # # Show the plots
# # plt.show()

# # # Check the length of each list in history_data
# # print(f"Length of loss: {len(history_data['loss'])}")
# # print(f"Length of accuracy: {len(history_data['accuracy'])}")
# # print(f"Length of val_loss: {len(history_data['val_loss'])}")
# # print(f"Length of val_accuracy: {len(history_data['val_accuracy'])}")




# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV files into DataFrames
# model_800 = pd.read_csv("/Users/divyansh/Downloads/scores_data_800_50.csv")  # Replace with actual path
# model_1200 = pd.read_csv("/Users/divyansh/Downloads/scores_data_best_50.csv")  # Replace with actual path

# # Assuming both files have columns 'Episode' and 'Score'
# # Ensure both CSVs have the same episode structure
# plt.figure(figsize=(10, 6))

# # Plotting the scores for both models
# plt.plot(model_800['Episode'], model_800['Score'], label='Model (800 Epochs)', color='blue', linewidth=2)
# plt.plot(model_1200['Episode'], model_1200['Score'], label='Model (1200 Epochs)', color='green', linewidth=2)

# # Adding title and labels
# plt.title('Performance of Models over 100 Episodes', fontsize=16)
# plt.xlabel('Episode Number', fontsize=12)
# plt.ylabel('Score', fontsize=12)

# # Adding a legend
# plt.legend()

# # Display the plot
# plt.grid(True)
# plt.tight_layout()  # Adjusts plot to fit labels
# plt.show()


import os
import sys
import numpy as np
import pandas as pd  # For saving to CSV
import matplotlib.pyplot as plt
from keras.models import load_model

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

def play(env, model, num_episodes=5):
    seed = 2000
    episode_rewards = []  # List to store the total reward for each episode
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        
        # Drop initial frames
        action0 = 0
        for i in range(5):
            obs, _, _, _, _ = env.step(action0)
        
        done = False
        total_reward = 0  # Initialize total reward for the episode
        
        while not done:
            # Reshape the observation to fit the model input
            p = model.predict(obs.reshape(1, 96, 96, 3))  # Adapt to your model
            action = np.argmax(p)  # Choose the action with the highest predicted value
            
            # Take a step in the environment with the predicted action
            obs, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward  # Accumulate the reward

            # Check if the episode is done
            done = terminated or truncated

            # Render the environment (if desired)
            env.render()

        episode_rewards.append(total_reward)  # Store the total reward of the episode
        print(f"Episode {episode + 1} Reward: {total_reward}")

    # Close the environment after all episodes
    env.close()
    
    # Return the episode rewards for further processing
    return episode_rewards

# Set up the environment
env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': "human"  # Set render_mode to human for interactive play
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Load your trained model
model = load_model('/Applications/Files/SEM_7/MAJOR/demo/GymCarCNNClassifier/models/model1_20_epochs_1e-2.h5')

# Play the game and collect episode rewards
episode_rewards = play(env, model, num_episodes=50)

# Save the episode rewards to a CSV file
episode_data = {'Episode': range(1, 51), 'Reward': episode_rewards}
df = pd.DataFrame(episode_data)

# Save to CSV file
csv_file = 'episode_rewards.csv'
# df.to_csv(csv_file, index=False)

print(f"Rewards saved to {csv_file}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), episode_rewards, marker='o', color='b', label='Reward')
plt.title('Episode Rewards')
plt.xlabel('Episode Number')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
