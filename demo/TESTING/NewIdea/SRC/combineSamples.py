import pickle
import numpy as np

def load_user_data(filenames):
    all_data = []  # List to hold data from all episodes

    for filename in filenames:
        with open(filename, 'rb') as f:
            episode_data = pickle.load(f)
            all_data.extend(episode_data)  # Combine data from this episode into all_data

    return all_data

def prepare_data_for_training(data):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # Separate the data into individual components
    for state, action, reward, next_state, done in data:
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    # Convert lists to numpy arrays for easier handling
    return (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(next_states),
        np.array(dones)
    )

# List of filenames for your collected episodes
filenames = ['/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data1.pkl', '/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data2.pkl', 
             '/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data3.pkl']

# Load and combine the user data
user_data = load_user_data(filenames)

# Prepare the data for training
states, actions, rewards, next_states, dones = prepare_data_for_training(user_data)

# Example: print the shapes of the resulting arrays
print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Rewards shape: {rewards.shape}")
print(f"Next States shape: {next_states.shape}")
print(f"Dones shape: {dones.shape}")
