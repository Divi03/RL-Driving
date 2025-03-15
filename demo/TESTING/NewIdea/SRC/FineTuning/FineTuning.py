import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load user data from multiple pickle files
def load_user_data(filenames):
    data = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            data.extend(pickle.load(f))
    return data

# Preprocess user data to prepare it for training
def preprocess_user_data(data):
    # states, actions, rewards, next_states = zip(*data)

    # Unpack the data into states, actions, rewards, next_states, and dones
    states, actions, rewards, next_states, dones = zip(*data)
    
    return states, actions

# Fine-tune the model on user data
def fine_tune_model(model, states, actions, epochs=10, batch_size=32):
    # One-hot encode the actions if necessary
    num_classes = model.output_shape[-1]
    actions = to_categorical(actions, num_classes=num_classes)

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(states, actions, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Main function
if __name__ == "__main__":
    # Load user data
    user_data_filenames = ['/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data1.pkl', '/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data2.pkl', '/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/DATA/user_data3.pkl']
    user_data = load_user_data(user_data_filenames)
    
    # Preprocess user data
    states, actions = preprocess_user_data(user_data)

    # Load the existing model
    model_path = '/Applications/Files/SEM_7/MAJOR/common/PRETRAINED_RL/model1_20_epochs_1e-2.h5'
    model = load_model(model_path)

    # Fine-tune the model on user data
    history = fine_tune_model(model, states, actions, epochs=10, batch_size=32)

    # Save the fine-tuned model
    model.save('fine_tuned_model.h5')

    print("Model fine-tuning complete and saved.")
