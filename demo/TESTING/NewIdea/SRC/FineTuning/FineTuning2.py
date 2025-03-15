# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
# from tensorflow.keras import optimizers
# import numpy as np

# # Assuming input_shape and num_classes are defined
# input_shape = (96, 96, 3)  # Adjust to your input shape
# num_classes = 5  # Adjust to your number of classes

# def ModelI(input_shape, num_classes, lr=0.0001):
#     model = Sequential()
#     model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(5,5)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#     model.add(BatchNormalization())
    
#     model.add(Conv2D(filters=64, kernel_size=(3,3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#     model.add(BatchNormalization())
    
#     model.add(Flatten())
    
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#     model.add(BatchNormalization())
    
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))

#     adam = optimizers.Adam(learning_rate=lr)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#     return model

# # Load the pretrained model (adjust the path as necessary)
# model1 = ModelI(input_shape, num_classes)
# model1.load_weights('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/MODELS/PRETRAINED_RL/model1_20_epochs_1e-2.h5')  # Load weights if saved separately

# # Freeze all layers except the last few
# for layer in model1.layers[:-3]:  # Adjust this number as needed
#     layer.trainable = False

# # Compile the model again after unfreezing layers
# model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# # Prepare your user data (ensure it's preprocessed correctly)
# # Assuming you have user_data as a NumPy array of images and user_labels as one-hot encoded labels
# user_data = np.array([...])  # Replace with your user data
# user_labels = np.array([...])  # Replace with your one-hot encoded labels

# # Define the callback for early stopping
# callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# # Fine-tune the model on user data
# history = model1.fit(user_data, user_labels, 
#                       validation_split=0.2,  # Use a portion of data for validation
#                       epochs=20,  # Adjust epochs as needed
#                       batch_size=32,  # Adjust batch size as needed
#                       callbacks=[callback1])

# # Save the fine-tuned model if needed
# model1.save('fine_tuned_model.h5')






import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras import optimizers
import numpy as np
import pickle

# Define input shape and number of classes
input_shape = (96, 96, 3)  # Adjust to your input shape
num_classes = 5  # Adjust to your number of classes

def ModelI(input_shape, num_classes, lr=0.0001):
    model = Sequential()
    
    # C1 Convolutional Layer
    model.add(Conv2D(filters=32, input_shape=input_shape, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile the model
    adam = optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

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
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # One-hot encode actions if not already done
    actions = tf.keras.utils.to_categorical(actions, num_classes)

    return states, actions, rewards, next_states, dones


# Load user data from .pkl files
user_data_files = ['/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data1.pkl', '/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data1.pkl', '/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data1.pkl']

# Load and prepare the user data
user_data = load_user_data(user_data_files)
states, actions, rewards, next_states, dones = prepare_data_for_training(user_data)

# Example: print the shapes of the resulting arrays
print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Rewards shape: {rewards.shape}")
print(f"Next States shape: {next_states.shape}")
print(f"Dones shape: {dones.shape}")

# Load the pretrained model
model1 = ModelI(input_shape, num_classes)
model1.load_weights('/Applications/Files/SEM_7/MAJOR/common/PRETRAINED_RL/model1_20_epochs_1e-2.h5')

# Freeze all layers except the last few
for layer in model1.layers[:-3]:  # Adjust this number to unfreeze more or fewer layers
    layer.trainable = False

# Compile the model again after unfreezing layers
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])


actions = tf.keras.utils.to_categorical(actions, num_classes)

# Define the callback for early stopping
callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Fine-tune the model on user data
history = model1.fit(states, actions, 
                      validation_split=0.2, 
                      epochs=20, 
                      batch_size=32, 
                      callbacks=[callback1])

# Save the fine-tuned model
model1.save('/Users/divyansh/Desktop/DEmo/TESTING/NewIdea/MODELS/FINETUNE/fine_tuned_model.h5')
