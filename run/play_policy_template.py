# import sys
# import numpy as np
# from keras.models import load_model
# # from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
# from gymnasium.wrappers import RecordVideo



# try:
#     import gymnasium as gym
# except ModuleNotFoundError:
#     print('gymnasium module not found. Try to install with')
#     print('pip install gymnasium[box2d]')
#     sys.exit(1)



# def play(env, model):
    
#     seed = 2000
#     obs, _ = env.reset(seed=seed)

#     # drop initial frames
#     action0 = 0
#     for i in range(50):
#         obs,_,_,_,_ = env.step(action0)
    
#     done = False
#     frames = []
#     # Use VideoRecorder for capturing frames
#     video_recorder = RecordVideo(env, "video/test.mp4")
#     # set the video velocity
#     video_recorder.frames_per_sec = 30
    
#     while not done:
#         p = model.predict(obs.reshape(1,96,96,3)) # adapt to your model
#         action = np.argmax(p)  # adapt to your model
#         print(action)
#         obs, _, terminated, truncated, _ = env.step(action)

#         done = terminated or truncated
#           # Capture the current frame
        
#         # Render the environment and record the frame
#         video_recorder.capture_frame()


#     # Save the recorded frames as a video
#     video_recorder.close()



# env_arguments = {
#     'domain_randomize': False,
#     'continuous': False,
#     'render_mode': "rgb_array"
# }

# env_name = 'CarRacing-v2'
# env = gym.make(env_name, **env_arguments)

# print("Environment:", env_name)
# print("Action space:", env.action_space)
# print("Observation space:", env.observation_space)


# # your trained model
# model = load_model('/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/MODELS/PRETRAINED_RL/model1_20_epochs_1e-2.h5')

# play(env, model)




# import sys
# import numpy as np
# from keras.models import load_model
# from gymnasium.wrappers import RecordVideo

# try:
#     import gymnasium as gym
# except ModuleNotFoundError:
#     print('gymnasium module not found. Try to install with')
#     print('pip install gymnasium[box2d]')
#     sys.exit(1)

# def play(env, model):
#     seed = 2000
#     obs, _ = env.reset(seed=seed)

#     # Drop initial frames
#     action0 = 0
#     for i in range(50):
#         obs, _, _, _, _ = env.step(action0)

#     done = False
#     # Use RecordVideo for capturing frames
#     video_recorder = RecordVideo(env, "./video/test.mp4")
#     # Set the video velocity
#     video_recorder.frames_per_sec = 30

#     while not done:
#         # Reshape the observation to fit the model input
#         p = model.predict(obs.reshape(1, 96, 96, 3))  # adapt to your model
#         action = np.argmax(p)  # adapt to your model
#         print(action)

#         # Take a step in the environment with the predicted action
#         obs, _, terminated, truncated, _ = env.step(action)

#         # Check if the episode is done
#         done = terminated or truncated

#     # Save the recorded frames as a video
#     video_recorder.close()

# # Set up the environment
# env_arguments = {
#     'domain_randomize': False,
#     'continuous': False,
#     'render_mode': "rgb_array"  # Ensure render_mode is rgb_array
# }

# env_name = 'CarRacing-v2'
# env = gym.make(env_name, **env_arguments)

# print("Environment:", env_name)
# print("Action space:", env.action_space)
# print("Observation space:", env.observation_space)

# # Load your trained model
# model = load_model('/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/MODELS/PRETRAINED_RL/model1_20_epochs_1e-2.h5')

# # Play the game
# play(env, model)





# import sys
# import numpy as np
# from keras.models import load_model

# try:
#     import gymnasium as gym
# except ModuleNotFoundError:
#     print('gymnasium module not found. Try to install with')
#     print('pip install gymnasium[box2d]')
#     sys.exit(1)

# def play(env, model):
#     seed = 2000
#     obs, _ = env.reset(seed=seed)

#     # Drop initial frames
#     action0 = 0
#     for i in range(50):
#         obs, _, _, _, _ = env.step(action0)

#     done = False

#     while not done:
#         # Reshape the observation to fit the model input
#         p = model.predict(obs.reshape(1, 96, 96, 3))  # adapt to your model
#         action = np.argmax(p)  # adapt to your model
#         print(action)

#         # Take a step in the environment with the predicted action
#         obs, _, terminated, truncated, _ = env.step(action)

#         # Check if the episode is done
#         done = terminated or truncated
        
#         # Render the environment in human mode
#         env.render()

#     # Close the environment after the game ends
#     env.close()

# # Set up the environment
# env_arguments = {
#     'domain_randomize': False,
#     'continuous': False,
#     'render_mode': "human"  # Set render_mode to human for interactive play
# }

# env_name = 'CarRacing-v2'
# env = gym.make(env_name, **env_arguments)

# print("Environment:", env_name)
# print("Action space:", env.action_space)
# print("Observation space:", env.observation_space)

# # Load your trained model
# model = load_model('/Applications/Files/SEM_7/MAJOR/demo/TESTING/NewIdea/MODELS/FINETUNE/fine_tuned_model_discrete3.h5')

# # Play the game
# play(env, model)





import sys
import numpy as np
from keras.models import load_model

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

def play(env, model):
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # Drop initial frames
    action0 = 0
    for i in range(50):
        obs, _, _, _, _ = env.step(action0)

    done = False

    while not done:
        # Reshape the observation to fit the model input
        p = model.predict(obs.reshape(1, 96, 96, 3))  # adapt to your model
        action = np.argmax(p)  # adapt to your model
        print(action)

        # Take a step in the environment with the predicted action
        obs, _, terminated, truncated, _ = env.step(action)

        # Check if the episode is done
        done = terminated or truncated
        
        # Render the environment in human mode
        env.render()

    # Close the environment after the game ends
    env.close()

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

# Play the game
play(env, model)

