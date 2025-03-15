# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# episodes = 5000
# start_score = -150
# end_score = 800

# # Generate synthetic episode scores (simulate improvement over episodes)
# np.random.seed(42)  # For reproducibility
# scores = np.linspace(start_score, end_score, episodes) + np.random.normal(0, 50, episodes)

# # Calculate running average (window size = 100)
# window_size = 100
# running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

# # Plotting the graph
# plt.figure(figsize=(10, 6))
# plt.plot(range(episodes), scores, label='Episode Score', alpha=0.7, color='blue')
# plt.plot(range(window_size - 1, episodes), running_avg, label='Running Average (100)', color='red', linestyle='--')

# # Adding titles and labels
# plt.title("Episode Scores and Running Average")
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for saving CSV files

# Initialize an empty list to store the rewards
rewards = []

# First 100 epochs: Rewards in range -100 to -10
rewards.append(np.linspace(-100, -10, 100))

# Next 200 epochs: Gradually increase rewards in range -50 to 0
rewards.append(np.linspace(-50, 0, 200))

# Next 200 epochs: Gradually increase rewards from 0 to 50
rewards.append(np.linspace(0, 50, 200))

# Next 500 epochs: Gradually increase rewards from 50 to 200
rewards.append(np.linspace(50, 200, 500))

# Next 500 epochs: Gradually increase rewards from 200 to 500
rewards.append(np.linspace(200, 500, 500))

# Next 1000 epochs: Gradually increase rewards from 500 to 700
rewards.append(np.linspace(500, 700, 1000))

# Next 2000 epochs: Gradually increase rewards from 700 to 900
rewards.append(np.linspace(700, 900, 2000))

# Next segment: Exponential growth towards 900
# We'll create exponential growth from the current value to 900, starting at 700

# Number of epochs in this final segment (you can adjust this)
exp_epochs = 5000  # This will determine how many epochs the exponential growth lasts

# Create an exponential growth for the last segment
x = np.linspace(0, 1, exp_epochs)  # x values between 0 and 1
exponential_growth = 700 + (900 - 700) * (np.exp(5 * x) - 1) / (np.exp(5) - 1)  # Exponential growth

rewards.append(exponential_growth)

# Convert the list of rewards to a single NumPy array
rewards = np.concatenate(rewards)

# Generate the x-axis data (steps) based on the length of rewards
steps = np.arange(0, len(rewards) * 1000, 1000)  # 1000 is the interval between epochs

# Add random noise with high magnitude to introduce variance (especially for later stages)
variance_magnitude = 300  # High magnitude for variance
variance_interval = 500  # Add variance every 500 steps

# Add noise in the later steps where high variance is required
for i in range(500, len(rewards), variance_interval):
    rewards[i:i + variance_interval] += np.random.randn(variance_interval) * variance_magnitude

# Ensure that rewards stay within the desired range (-100 to 900)
rewards = np.clip(rewards, -100, 900)

# Add small noise for general fluctuations (across all steps)
rewards += np.random.randn(len(rewards)) * 20


# Calculate and plot the moving average to smooth the fluctuations
window_size = 50  # Smaller window for moving average
moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

# Create the plot
plt.figure(figsize=(12, 6))
# plt.plot(steps, rewards, color='red', label='Reward')
plt.plot(steps[window_size - 1:], moving_avg, color='blue', label='Moving Average', linewidth=1)
plt.grid(axis='y', linestyle='--')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('Reward vs. Epoch with Exponential Growth and Moving Average', fontsize=16)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()



# Create a DataFrame to store the steps and rewards
data = pd.DataFrame({
    'Epoch': steps[window_size - 1:],
    'Reward': moving_avg
})

# Save the DataFrame to a CSV file
data.to_csv('rewards_vs_epochs.csv', index=False)




# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = '/Applications/Files/SEM_7/MAJOR/rewards_vs_epochs.csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Check the structure of the dataframe
# print(df.head())  # To inspect the first few rows of your CSV

# # Calculate moving average if it's not already present
# window_size = 250  # You can adjust this window size depending on your data
# df['MovingAvg'] = df['Reward'].rolling(window=window_size).mean()

# # Plotting the data
# plt.figure(figsize=(10, 6))


# # Plot Reward
# plt.plot(df['Epoch'], df['Reward'], label='Reward', color='blue', linestyle='-')

# # Plot Moving Average
# plt.plot(df['Epoch'], df['MovingAvg'], label=f'Moving Avg (Window={window_size})', color='red', linestyle='-')

# # Labels and title
# plt.title('Rewards and Moving Average')
# plt.xlabel('Epoch')
# plt.ylabel('Reward')
# plt.legend(loc='best')

# # Show the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()