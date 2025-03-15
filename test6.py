# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Parameters
# # episodes = 5000

# # # Generate synthetic episode scores with exponential growth
# # np.random.seed(42)  # For reproducibility
# # scores = []

# # # First 1000 episodes: Scores between -150 and 150 with small variances
# # for i in range(1000):
# #     score = np.random.uniform(-150, 150) + np.random.normal(0, 20)
# #     scores.append(score)

# # # Next 1000 episodes: Scores between 150 and 300 with random fluctuations
# # for i in range(1000, 2000):
# #     score = np.random.uniform(150, 300) + np.random.normal(0, 25)
# #     scores.append(score)

# # # Next 1000 episodes: Scores between 300 and 500 with increasing variances
# # for i in range(2000, 3000):
# #     score = np.random.uniform(300, 500) + np.random.normal(0, 30)
# #     scores.append(score)

# # # Last 2000 episodes: Exponentially increasing towards 820
# # for i in range(3000, 5000):
# #     # Exponentially increase towards 820
# #     growth_factor = 1 + (i - 3000) / 2000  # Gradually increasing factor
# #     score = min(820, scores[-1] * growth_factor + np.random.normal(0, 20))
# #     scores.append(score)

# # # Convert the list to a numpy array
# # scores = np.array(scores)

# # # Calculate running average (window size = 100)
# # window_size = 100
# # running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

# # # Plotting the graph
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(episodes), scores, label='Episode Score', alpha=0.7, color='blue')
# # plt.plot(range(window_size - 1, episodes), running_avg, label='Running Average (100)', color='red', linestyle='--')

# # # Adding titles and labels
# # plt.title("Episode Scores and Running Average (Exponential Growth)")
# # plt.xlabel("Episode")
# # plt.ylabel("Score")
# # plt.legend()
# # plt.grid(alpha=0.3)
# # plt.show()



# # 
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# episodes = 5000

# # Generate synthetic episode scores with the specified structure
# np.random.seed(42)  # For reproducibility
# scores = []

# # First 1000 episodes: Narrow range with small variance (between -50 to 50)
# for i in range(1000):
#     score = np.random.uniform(-50, 50) + np.random.normal(0, 15)
#     scores.append(score)

# # Next 1000 episodes: Wider range, allowing some negative values (between -100 to 150)
# for i in range(1000, 2000):
#     score = np.random.uniform(-100, 150) + np.random.normal(0, 20)
#     scores.append(score)

# # Next 1000 episodes: Even wider range (between -150 to 200) with moderate variance
# for i in range(2000, 3000):
#     score = np.random.uniform(-150, 200) + np.random.normal(0, 30)
#     scores.append(score)

# # Last 2000 episodes: Gradually increasing towards 820, with decreasing variance
# for i in range(3000, 5000):
#     # Gradual increase toward 820, with small fluctuations but allowing for large range
#     if i < 4000:
#         score = np.random.uniform(200, 400) + np.random.normal(0, 25)  # Larger variance
#     else:
#         score = np.random.uniform(400, 800) + np.random.normal(0, 15)  # Decreasing variance
    
#     # Ensure the score doesn't exceed 820
#     scores.append(min(score, 820))

# # Convert the list to a numpy array
# scores = np.array(scores)

# # Calculate running average (window size = 100)
# window_size = 100
# running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

# # Plotting the graph
# plt.figure(figsize=(10, 6))
# plt.plot(range(episodes), scores, label='Episode Score', alpha=0.7, color='blue')
# plt.plot(range(window_size - 1, episodes), running_avg, label='Running Average (100)', color='red', linestyle='--')

# # Adding titles and labels
# plt.title("Episode Scores and Running Average (Controlled Growth)")
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# episodes = 5000

# # Generate synthetic episode scores with the specified structure
# np.random.seed(42)  # For reproducibility
# scores = []

# # First 1000 episodes: Narrow range with small variance (between -50 to 50)
# for i in range(1000):
#     score = np.random.uniform(-50, 50) + np.random.normal(0, 15)
#     scores.append(score)

# # Next 1000 episodes: Wider range, allowing some negative values (between -100 to 150)
# for i in range(1000, 2000):
#     score = np.random.uniform(-100, 150) + np.random.normal(0, 20)
#     scores.append(score)

# # Next 1000 episodes: Even wider range (between -150 to 200) with moderate variance
# for i in range(2000, 3000):
#     score = np.random.uniform(-150, 200) + np.random.normal(0, 30)
#     scores.append(score)

# # Last 2000 episodes: Gradually increasing towards 820, with decreasing variance
# for i in range(3000, 4000):
#     # Gradual increase toward 820, with very small fluctuations and smaller range
#     score = np.random.uniform(200, 400) + np.random.normal(0, 10)  # Narrower range, very small variance
#     scores.append(min(score, 820))

# for i in range(4000, 5000):
#     # Even narrower range with very small fluctuations, smooth increase
#     score = np.random.uniform(300, 500) + np.random.normal(0, 5)  # Very small variance
#     scores.append(min(score, 820))  # Ensure the score does not exceed 820

# # Convert the list to a numpy array
# scores = np.array(scores)

# # Calculate running average (window size = 100)
# window_size = 100
# running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

# # Plotting the graph
# plt.figure(figsize=(10, 6))
# plt.plot(range(episodes), scores, label='Episode Score', alpha=0.7, color='blue')
# plt.plot(range(window_size - 1, episodes), running_avg, label='Running Average (100)', color='red', linestyle='--')

# # Adding titles and labels
# plt.title("Episode Scores and Running Average (Controlled Growth)")
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Parameters
episodes = 5000

# Generate synthetic episode scores with the specified structure
np.random.seed(42)  # For reproducibility
scores = []

# First 1000 episodes: Adjusted starting range between -50 and 10
for i in range(1000):
    score = np.random.uniform(-50, 100) + np.random.normal(0, 20)
    scores.append(score)

# Next 1000 episodes: Shift range to between -30 and 50
for i in range(1000, 2000):
    score = np.random.uniform(10, 250) + np.random.normal(0, 25)
    scores.append(score)

# Next 1000 episodes (2000-3000): Shift range to between -10 and 100, no negative values below -10
for i in range(2000, 3000):
    score = np.random.uniform(50, 350) + np.random.normal(0, 25)
    scores.append(score)

# Last 2000 episodes: Gradual increase towards 820, with positive spikes and rare negative values
for i in range(3000, 4000):
    # Gradual increase towards 820, with small positive spikes
    if np.random.rand() < 0.1:  # Occasional positive spike (low frequency)
        score = np.random.uniform(100, 200) + np.random.normal(0, 10)
    else:
        score = np.random.uniform(200, 500) + np.random.normal(0, 5)  # Smooth growth
    scores.append(min(score, 820))  # Ensure the score doesn't exceed 820

# Gradually reach 820, with controlled variance and occasional positive spikes
for i in range(4000, 5000):
    # Gradual increase towards 820
    if np.random.rand() < 0.05:  # Even rarer positive spike
        score = np.random.uniform(300, 500) + np.random.normal(0, 5)  # Small positive spike
    else:
        # Gradual increase towards 820 with small variance
        score = np.random.uniform(200, 800) + np.random.normal(0, 3)  # Gradual smooth growth
    scores.append(min(score, 800))  # Ensure the score doesn't exceed 820

# Convert the list to a numpy array
scores = np.array(scores)

# Calculate running average (window size = 100)
window_size = 50
running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(range(episodes), scores, label='Episode Score', alpha=0.7, color='blue')
plt.plot(range(window_size - 1, episodes), running_avg, label='Running Average (100)', color='red', linestyle='--')

# Adding titles and labels
plt.title("Episode Scores and Running Average")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
