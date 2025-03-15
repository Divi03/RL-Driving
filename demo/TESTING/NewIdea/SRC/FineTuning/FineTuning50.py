import pickle
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
import gym

# Load the discrete user data from the pickle file
def load_user_data(filename='/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data_discrete.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Prepare the data for training
def prepare_data_for_training(data):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for state, action, reward, next_state, done in data:
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions).reshape(-1, 1)  # Shape (n_samples, 1)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # One-hot encode the actions (since we are using discrete actions)
    encoder = OneHotEncoder(sparse=False)
    actions_encoded = encoder.fit_transform(actions)  # Shape (n_samples, n_actions)

    return states, actions_encoded, rewards, next_states, dones

# Define the model architecture (same as the original model)
def ModelI(input_shape, num_classes, lr=0.0001):
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=64, kernel_size=(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

# Play and record episodes using the model
def play_and_record_episodes(env, model, num_episodes=5):
    data = []

    for episode in range(num_episodes):
        observation = env.reset()
        
        # Extract the observation if it's a tuple
        if isinstance(observation, tuple):
            observation = observation[0]
        
        done = False
        while not done:
            state = observation / 255.0  # Normalize the observation
            
            # Model prediction (assuming model expects a batch of data, hence the [np.newaxis, ...])
            action_probs = model.predict(state[np.newaxis, ...])
            action = np.argmax(action_probs)
            
            next_observation, reward, done, info = env.step(action)

            # Extract next observation if it's a tuple
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            
            # Record state, action, reward, next_state, done
            data.append((observation, action, reward, next_observation, done))
            
            # Move to the next observation
            observation = next_observation

    return data

# Iteratively train the model
def iterative_training(env, model, initial_data, num_iterations=50, num_episodes_per_iteration=5):
    # Prepare the initial data
    states, actions_encoded, rewards, next_states, dones = prepare_data_for_training(initial_data)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Fine-tune the model on the initial and generated data
        history = model.fit(states, actions_encoded,
                            validation_split=0.05,
                            epochs=20,
                            batch_size=32)

        # Use the newly trained model to play and record episodes
        new_data = play_and_record_episodes(env, model, num_episodes=num_episodes_per_iteration)
        
        # Append new data to the existing data
        new_states, new_actions_encoded, new_rewards, new_next_states, new_dones = prepare_data_for_training(new_data)
        
        states = np.vstack((states, new_states))
        actions_encoded = np.vstack((actions_encoded, new_actions_encoded))
        rewards = np.hstack((rewards, new_rewards))
        next_states = np.vstack((next_states, new_next_states))
        dones = np.hstack((dones, new_dones))

    # Save the final model after 50 iterations
    model.save('fine_tuned_model_iterative.h5')

# Load and preprocess the discrete user data
user_data = load_user_data('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data_discrete.pkl')

# Load the pretrained model and its weights
input_shape = (96, 96, 3)
num_classes = 5
model1 = ModelI(input_shape, num_classes)
model1.load_weights('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/MODELS/PRETRAINED_RL/model1_20_epochs_1e-2.h5')  # Replace with the correct path

# Freeze layers except the last few
for layer in model1.layers[:-3]:
    layer.trainable = False

# Recompile the model after unfreezing certain layers
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Initialize the environment (e.g., from OpenAI Gym)
env = gym.make('CarRacing-v2')  # Replace with the correct environment

# Run the iterative training process
iterative_training(env, model1, user_data, num_iterations=50, num_episodes_per_iteration=5)
