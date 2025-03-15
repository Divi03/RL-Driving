import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers
from sklearn.preprocessing import OneHotEncoder

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

# Load and preprocess the discrete user data
user_data = load_user_data('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/DATA/user_data_discrete.pkl')
states, actions_encoded, rewards, next_states, dones = prepare_data_for_training(user_data)

# Assuming the input shape is (96, 96, 3) and there are 5 discrete actions
input_shape = (96, 96, 3)
num_classes = 5  # Discrete actions: 0, 1, 2, 3, 4

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

# Load the pretrained model and its weights
model1 = ModelI(input_shape, num_classes)
model1.load_weights('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/MODELS/PRETRAINED_RL/model1_20_epochs_1e-2.h5')  # Replace with the correct path

# Freeze layers except the last few
for layer in model1.layers[:-3]:  # Keep the last 3 layers trainable
    layer.trainable = False

# Recompile the model after unfreezing certain layers
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Fine-tune the model on the discrete user data
callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model1.fit(states, actions_encoded, 
                      validation_split=0.05,  # Use 20% // 5% of data for validation
                      epochs=50,  # Adjust epochs as needed
                      batch_size=32,  # Adjust batch size as needed
                      callbacks=[callback1])

# Save the fine-tuned model
model1.save('fine_tuned_model_discrete3.h5')
