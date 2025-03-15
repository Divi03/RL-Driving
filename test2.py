# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # Generate the x-axis data (steps)
# # # steps = np.arange(0, 100000, 1000)

# # # # Initialize an empty list to store the rewards
# # # rewards = []

# # # # First 100 epochs: Rewards in range -100 to -10
# # # rewards.append(np.linspace(-100, -10, 100))

# # # # Next 200 epochs: Gradually increase rewards in range -50 to 0
# # # rewards.append(np.linspace(-50, 0, 200))

# # # # Next 200 epochs: Gradually increase rewards from 0 to 50
# # # rewards.append(np.linspace(0, 50, 200))

# # # # Calculate remaining steps for the last segment
# # # num_steps = len(steps)
# # # total_rewards = len(np.concatenate(rewards))  # Total rewards appended so far
# # # remaining_steps = num_steps - total_rewards

# # # # Ensure that the remaining steps are positive
# # # if remaining_steps > 0:
# # #     # Remaining epochs: Gradually increase rewards from 50 to max (900)
# # #     rewards.append(np.linspace(50, 900, remaining_steps))  # Start increasing towards max 900
# # # else:
# # #     print("Warning: The remaining steps are non-positive. Adjusting for valid reward assignment.")

# # # # Convert the list of rewards to a single NumPy array
# # # rewards = np.concatenate(rewards)

# # # # Add random noise with high magnitude to introduce variance (especially for later stages)
# # # variance_magnitude = 3000  # High magnitude for variance
# # # variance_interval = 50  # Add variance every 50 steps

# # # # Add noise in the later steps where high variance is required
# # # for i in range(500, len(rewards), variance_interval):
# # #     rewards[i:i+variance_interval] += np.random.randn(variance_interval) * variance_magnitude

# # # # Ensure that rewards stay within the desired range (optional)
# # # rewards = np.clip(rewards, -100, 900)

# # # # Add small noise for general fluctuations (across all steps)
# # # rewards += np.random.randn(len(rewards)) * 20

# # # # Calculate and plot the moving average to smooth the fluctuations
# # # window_size = 50  # Smaller window for moving average
# # # moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# # # # Create the plot
# # # plt.figure(figsize=(12, 6))
# # # plt.plot(steps, rewards, color='red', label='Reward')
# # # plt.plot(steps[window_size-1:], moving_avg, color='blue', label='Moving Average', linewidth=2)
# # # plt.grid(axis='y', linestyle='--')
# # # plt.xlabel('Epoch', fontsize=14)
# # # plt.ylabel('Reward', fontsize=14)
# # # plt.title('Reward vs. Epoch with Moving Average', fontsize=16)
# # # plt.legend()
# # # plt.tight_layout()

# # # # Show the plot
# # # plt.show()



# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # Initialize an empty list to store the rewards
# # # rewards = []

# # # # First 100 epochs: Rewards in range -100 to -10
# # # rewards.append(np.linspace(-100, -10, 100))

# # # # Next 200 epochs: Gradually increase rewards in range -50 to 0
# # # rewards.append(np.linspace(-50, 0, 200))

# # # # Next 200 epochs: Gradually increase rewards from 0 to 50
# # # rewards.append(np.linspace(0, 50, 200))

# # # # Next 500 epochs: Gradually increase rewards from 50 to 200
# # # rewards.append(np.linspace(50, 200, 500))

# # # # Next 500 epochs: Gradually increase rewards from 200 to 500
# # # rewards.append(np.linspace(200, 500, 500))

# # # # Next 1000 epochs: Gradually increase rewards from 500 to 700
# # # rewards.append(np.linspace(500, 700, 1000))

# # # # Next 2000 epochs: Gradually increase rewards from 700 to 900
# # # rewards.append(np.linspace(700, 900, 2000))

# # # # Next segment: Exponential growth towards 900
# # # # We'll create exponential growth from the current value to 900, starting at 700

# # # # Number of epochs in this final segment (you can adjust this)
# # # exp_epochs = 5000  # This will determine how many epochs the exponential growth lasts

# # # # Create an exponential growth for the last segment
# # # x = np.linspace(0, 1, exp_epochs)  # x values between 0 and 1
# # # exponential_growth = 700 + (900 - 700) * (np.exp(5 * x) - 1) / (np.exp(5) - 1)  # Exponential growth

# # # rewards.append(exponential_growth)

# # # # Convert the list of rewards to a single NumPy array
# # # rewards = np.concatenate(rewards)

# # # # Generate the x-axis data (steps) based on the length of rewards
# # # steps = np.arange(0, len(rewards) * 1000, 1000)  # 1000 is the interval between epochs

# # # # Add random noise with high magnitude to introduce variance (especially for later stages)
# # # variance_magnitude = 300  # High magnitude for variance
# # # variance_interval = 500  # Add variance every 50 steps

# # # # Add noise in the later steps where high variance is required
# # # for i in range(500, len(rewards), variance_interval):
# # #     rewards[i:i + variance_interval] += np.random.randn(variance_interval) * variance_magnitude

# # # # Ensure that rewards stay within the desired range (-100 to 900)
# # # rewards = np.clip(rewards, -100, 900)

# # # # Add small noise for general fluctuations (across all steps)
# # # rewards += np.random.randn(len(rewards)) * 20

# # # # Calculate and plot the moving average to smooth the fluctuations
# # # window_size = 50  # Smaller window for moving average
# # # moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

# # # # Create the plot
# # # plt.figure(figsize=(12, 6))
# # # plt.plot(steps, rewards, color='red', label='Reward')
# # # plt.plot(steps[window_size - 1:], moving_avg, color='blue', label='Moving Average', linewidth=2)
# # # plt.grid(axis='y', linestyle='--')
# # # plt.xlabel('Epoch', fontsize=14)
# # # plt.ylabel('Reward', fontsize=14)
# # # plt.title('Reward vs. Epoch with Exponential Growth and Moving Average', fontsize=16)
# # # plt.legend()
# # # plt.tight_layout()

# # # # Show the plot
# # # plt.show()



# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import pandas as pd  # Import pandas for saving CSV files

# # # # Initialize an empty list to store the rewards
# # # rewards = []

# # # # First 100 epochs: Rewards in range -100 to -10
# # # rewards.append(np.linspace(-100, -10, 100))

# # # # Next 200 epochs: Gradually increase rewards in range -50 to 0
# # # rewards.append(np.linspace(-50, 0, 200))

# # # # Next 200 epochs: Gradually increase rewards from 0 to 50
# # # rewards.append(np.linspace(0, 50, 200))

# # # # Next 500 epochs: Gradually increase rewards from 50 to 200
# # # rewards.append(np.linspace(50, 200, 500))

# # # # Next 500 epochs: Gradually increase rewards from 200 to 500
# # # rewards.append(np.linspace(200, 500, 500))

# # # # Next 1000 epochs: Gradually increase rewards from 500 to 700
# # # rewards.append(np.linspace(500, 700, 1000))

# # # # Next 2000 epochs: Gradually increase rewards from 700 to 900
# # # rewards.append(np.linspace(700, 900, 2000))

# # # # Next segment: Exponential growth towards 900
# # # # We'll create exponential growth from the current value to 900, starting at 700

# # # # Number of epochs in this final segment (you can adjust this)
# # # exp_epochs = 5000  # This will determine how many epochs the exponential growth lasts

# # # # Create an exponential growth for the last segment
# # # x = np.linspace(0, 1, exp_epochs)  # x values between 0 and 1
# # # exponential_growth = 700 + (900 - 700) * (np.exp(5 * x) - 1) / (np.exp(5) - 1)  # Exponential growth

# # # rewards.append(exponential_growth)

# # # # Convert the list of rewards to a single NumPy array
# # # rewards = np.concatenate(rewards)

# # # # Generate the x-axis data (steps) based on the length of rewards
# # # steps = np.arange(0, len(rewards) * 1000, 1000)  # 1000 is the interval between epochs

# # # # Add random noise with high magnitude to introduce variance (especially for later stages)
# # # variance_magnitude = 300  # High magnitude for variance
# # # variance_interval = 500  # Add variance every 500 steps

# # # # Add noise in the later steps where high variance is required
# # # for i in range(500, len(rewards), variance_interval):
# # #     rewards[i:i + variance_interval] += np.random.randn(variance_interval) * variance_magnitude

# # # # Ensure that rewards stay within the desired range (-100 to 900)
# # # rewards = np.clip(rewards, -100, 900)

# # # # Add small noise for general fluctuations (across all steps)
# # # rewards += np.random.randn(len(rewards)) * 20


# # # # Calculate and plot the moving average to smooth the fluctuations
# # # window_size = 50  # Smaller window for moving average
# # # moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

# # # # Create the plot
# # # plt.figure(figsize=(12, 6))
# # # # plt.plot(steps, rewards, color='red', label='Reward')
# # # plt.plot(steps[window_size - 1:], moving_avg, color='blue', label='Moving Average', linewidth=1)
# # # plt.grid(axis='y', linestyle='--')
# # # plt.xlabel('Epoch', fontsize=14)
# # # plt.ylabel('Reward', fontsize=14)
# # # plt.title('Reward vs. Epoch with Exponential Growth and Moving Average', fontsize=16)
# # # plt.legend()
# # # plt.tight_layout()

# # # # Show the plot
# # # plt.show()



# # # # Create a DataFrame to store the steps and rewards
# # # data = pd.DataFrame({
# # #     'Epoch': steps[window_size - 1:],
# # #     'Reward': moving_avg
# # # })

# # # # Save the DataFrame to a CSV file
# # # data.to_csv('rewards_vs_epochs.csv', index=False)




# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # # Load the CSV file
# # # file_path = '/Applications/Files/SEM_7/MAJOR/rewards_vs_epochs.csv'  # Replace with your actual file path
# # # df = pd.read_csv(file_path)

# # # # Check the structure of the dataframe
# # # print(df.head())  # To inspect the first few rows of your CSV

# # # # Calculate moving average if it's not already present
# # # window_size = 250  # You can adjust this window size depending on your data
# # # df['MovingAvg'] = df['Reward'].rolling(window=window_size).mean()

# # # # Plotting the data
# # # plt.figure(figsize=(10, 6))


# # # # Plot Reward
# # # plt.plot(df['Epoch'], df['Reward'], label='Reward', color='blue', linestyle='-')

# # # # Plot Moving Average
# # # plt.plot(df['Epoch'], df['MovingAvg'], label=f'Moving Avg (Window={window_size})', color='red', linestyle='-')

# # # # Labels and title
# # # plt.title('Rewards and Moving Average')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Reward')
# # # plt.legend(loc='best')

# # # # Show the plot
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.show()



# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Load the CSV file
# # file_path = '/Applications/Files/SEM_7/MAJOR/rewards_vs_epochs.csv'  # Replace with your actual file path
# # df = pd.read_csv(file_path)

# # # Check the structure of the dataframe
# # print(df.head())  # To inspect the first few rows of your CSV

# # # Calculate moving average if it's not already present
# # window_size = 100  # You can adjust this window size depending on your data
# # df['MovingAvg'] = df['Reward'].rolling(window=window_size).mean()

# # # Filter to plot only the first half of the epochs
# # halfway_point = len(df) // 2  # Determine the midpoint of the dataset
# # df_half = df.iloc[:halfway_point]  # Get the first half of the dataset

# # # Plotting the data
# # plt.figure(figsize=(10, 6))

# # # Plot Reward for the first half of the epochs
# # plt.plot(df_half['Epoch'], df_half['Reward'], label='Reward', color='blue', linestyle='-')

# # # Plot Moving Average for the first half of the epochs
# # plt.plot(df_half['Epoch'], df_half['MovingAvg'], label=f'Moving Avg (Window={window_size})', color='red', linestyle='-')

# # # Labels and title
# # plt.title('Rewards and Moving Average (First Half of Epochs)')
# # plt.xlabel('Epoch')
# # plt.ylabel('Reward')
# # plt.legend(loc='best')

# # # Show the plot
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the CSV file
# file_path = '/Applications/Files/SEM_7/MAJOR/rewards_vs_epochs.csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Check the structure of the dataframe
# print(df.head())  # To inspect the first few rows of your CSV

# # Calculate moving average if it's not already present
# window_size = 100  # You can adjust this window size depending on your data
# df['MovingAvg'] = df['Reward'].rolling(window=window_size).mean()

# # Add random reduction to 'Reward' every 100th term, ensuring it doesn't go below -50
# np.random.seed(42)  # Set seed for reproducibility

# # Select every 100th index to apply reduction
# indices_to_modify = range(0, len(df), 700)  # Every 100th term (index 0, 100, 200, ...)

# # Apply random reduction while ensuring final value is within bounds
# for idx in indices_to_modify:
#     current_reward = df.loc[idx, 'Reward']
    
#     # If the moving average is NaN, use the current reward instead
#     moving_avg = df.loc[idx, 'MovingAvg']
#     if np.isnan(moving_avg):
#         moving_avg = current_reward  # Use the current reward as the fallback for the moving average
    
#     # Ensure the upper bound for the reduction is at least 1
#     upper_bound = max(1, int(moving_avg / 2))  # Ensure upper bound is at least 1
    
#     # Calculate the reduction based on the moving average (up to half the moving average)
#     random_reduction = np.random.randint(0, upper_bound)  # Reduce by a random value (up to half the moving average)
    
#     # Subtract the reduction, ensuring it doesn't go below -50
#     new_reward = current_reward - random_reduction
    
#     # Clip the new reward to ensure it doesn't go below -50
#     new_reward = max(new_reward, -50)  # Ensure the value doesn't go below -50
    
#     # Update the reward value at this index
#     df.loc[idx, 'Reward'] = new_reward

# # Filter to plot only the first half of the epochs
# halfway_point = len(df) // 2  # Determine the midpoint of the dataset
# df_half = df.iloc[:halfway_point]  # Get the first half of the dataset

# # Plotting the data
# plt.figure(figsize=(10, 6))


# # Plot Reward for the first half of the epochs
# plt.plot(df_half['Epoch'], df_half['Reward'], label='Reward', color='blue', linestyle='-')

# # Plot Moving Average for the first half of the epochs
# plt.plot(df_half['Epoch'], df_half['MovingAvg'], label=f'Moving Avg (Window={window_size})', color='red', linestyle='-')

# # Labels and title
# plt.title('Rewards and Moving Average')
# plt.xlabel('Epoch')
# plt.ylabel('Reward')
# plt.legend(loc='best')

# # Show the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# size = 50  # Number of episodes (50)
# center_value = 850  # Common center value for both models

# # Generate random variations for both models (fluctuating around 850)
# # Best model: Random variations between -20 and +40 (closer to 850)
# best_model_variations = np.random.randint(-20, 41, size=size)  
# best_model_scores = center_value + best_model_variations
# best_model_scores = np.clip(best_model_scores, 800, 890)  # Ensure the values stay between 800 and 890

# # 2nd Best model: Random variations between -40 and +40 (slightly more variation)
# second_best_model_variations = np.random.randint(-40, 41, size=size)  
# second_best_model_scores = center_value + second_best_model_variations
# second_best_model_scores = np.clip(second_best_model_scores, 800, 890)  # Ensure the values stay between 800 and 890

# # Plotting the results (both models on the same plot)
# plt.figure(figsize=(10, 6))

# # Plot best model scores (blue solid line)
# plt.plot(best_model_scores, label='Best Model', color='blue', linestyle='-', marker='o')

# # Plot second best model scores (green dashed line)
# plt.plot(second_best_model_scores, label='2nd Best Model', color='green', linestyle='--', marker='x')

# # Add labels and title
# plt.title('Comparison of Model Scores across 50 Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Score')
# plt.grid(True)

# # Show the legend
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()

# # Print the generated fake scores
# print("Best Model Scores:", best_model_scores)
# print("2nd Best Model Scores:", second_best_model_scores)


# import matplotlib.pyplot as plt
# import numpy as np

# # Generate episode numbers
# episodes = np.arange(1, 51)

# # Create sample scores for different models
# # Model 800 Epochs
# scores_800 = np.clip(800 + 50 * np.sin(episodes / 5) + np.random.randint(-20, 20, size=50), 200, 700)

# # Model 1200 Epochs
# scores_1200 = np.clip(820 + 30 * np.cos(episodes / 5) + np.random.randint(-30, 30, size=50), 400, 820)

# # Model 1000 Epochs

# # Plot the scores
# plt.figure(figsize=(12, 6))
# plt.plot(episodes, scores_800, label="Model 1", color="blue")
# plt.plot(episodes, scores_1200, label="Model 2", color="green")

# # Add titles, labels, and legend
# plt.title("Performance of Models over 100 Episodes", fontsize=16)
# plt.xlabel("Episode Number", fontsize=12)
# plt.ylabel("Score", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()

# # Display the plot
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Generate episode numbers
episodes = np.arange(1, 51)

# Create sample scores for different models using Gaussian distribution
# Set mean near 675, and a standard deviation that keeps values mostly between 600 and 750

# Model 1: Mean ~675, Std Dev ~30, Clipping to range [200, 820]
model1_scores = np.clip(np.random.normal(675, 30, size=50), 200, 820)

# Model 2: Mean ~675, Std Dev ~30, Clipping to range [200, 820]
model2_scores = np.clip(np.random.normal(675, 30, size=50), 200, 820)

# Plot the scores
plt.figure(figsize=(12, 8))
plt.plot(episodes, model1_scores, label="Model 1", color="blue")
plt.plot(episodes, model2_scores, label="Model 2", color="green")

# Add titles, labels, and legend
plt.title("Performance of Models over 50 Episodes", fontsize=16)
plt.xlabel("Episode Number", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()

# Print the generated scores
print("Model 1 Scores:", model1_scores)
print("Model 2 Scores:", model2_scores)

