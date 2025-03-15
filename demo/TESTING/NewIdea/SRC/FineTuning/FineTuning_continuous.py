import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers

# Assuming input_shape and num_classes are defined
input_shape = (96, 96, 3)  # Adjust to your input shape
num_actions = 3  # For continuous action space

def ModelI(input_shape, num_actions, lr=0.0001):
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
    
    # Output Layer for Continuous Actions
    model.add(Dense(num_actions))  # Outputs 3 continuous values
    model.add(Activation('tanh'))   # Use 'tanh' to output values between -1 and 1

    adam = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])  # Using mean squared error for continuous outputs

    return model

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

# Load your user data
user_data_files = ['/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data1.pkl', '/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data2.pkl', '/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data3.pkl']

user_data = load_user_data(user_data_files)

# Prepare the data for training
states, actions, rewards, next_states, dones = prepare_data_for_training(user_data)

# Example: print the shapes of the resulting arrays
print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Rewards shape: {rewards.shape}")
print(f"Next States shape: {next_states.shape}")
print(f"Dones shape: {dones.shape}")

# Create the model
model1 = ModelI(input_shape, num_actions)

# Fine-tune the model on user data
callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Fine-tune the model
history = model1.fit(states, actions, 
                      validation_split=0.5,  # Use a portion of data for validation
                      epochs=10,  # Adjust epochs as needed
                      batch_size=32,  # Adjust batch size as needed
                      callbacks=[callback1])

# Save the fine-tuned model
model1.save('fine_tuned_model_continuous.h5')
