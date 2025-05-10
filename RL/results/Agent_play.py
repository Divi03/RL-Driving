import os
import torch
import gymnasium as gym
import numpy as np
import time
from torch.distributions import Beta
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    """
    Convolutional Neural Network for PPO
    """

    def __init__(self, img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


img_stack=4

transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), 
                       ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])
GAMMA=0.99
EPOCH= 8 # beter than 10
MAX_SIZE = 2000 ## CUDA out of mem for max_size=10000
BATCH=128 
EPS=0.1
LEARNING_RATE = 0.001 # bettr than 0.005 or 0.002 
action_repeat = 10

def rgb2gray(rgb, norm=True):
    # Convert RGB to grayscale using the standard formula
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # RGB to grayscale
    if norm:
        # Normalize the grayscale image to range [-1, 1]
        gray = gray / 128.0 - 1.0
    return gray
class Agent():
    """ Agent for training """
    
    def __init__(self, device):
        self.training_step = 0
        self.net = Net(img_stack).double().to(device)
        self.buffer = np.empty(MAX_SIZE, dtype=transition)
        self.counter = 0
        self.device = device
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)  ## lr=1e-3

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp


    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == MAX_SIZE:
            self.counter = 0
            return True
        else:
            return False


agent = Agent('cpu')


class Wrapper:
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, env, img_stack=4, action_repeat=4):
        self.env = env
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.die = False
        self.stack = []
        self.av_r = self.reward_memory()

    def reset(self):
        self.counter = 0
        self.die = False
        img_rgb, _ = self.env.reset()  # Correct unpacking
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # Stack the initial frames
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):  # Loop for action repeats
            action[0] = np.clip(action[0], -1.0, 1.0)  # Steering (-1 to 1)
            action[1] = np.clip(action[1], 0.0, 1.0)   # Gas (0 to 1)
            action[2] = np.clip(action[2], 0.0, 1.0)   # Braking (0 to 1)
            img_rgb, reward, done, truncated, info = self.env.step(action)

            # Apply penalties and rewards
            if self.die:
                reward += 100
            if np.mean(img_rgb[:, :, 1]) > 185.0:  # Green penalty
                reward -= 0.05

            total_reward += reward

            # Calculate rolling average reward to decide on episode termination
            done = True if self.av_r(reward) <= -0.1 else done
            if done or self.die:
                break

        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack

        return np.array(self.stack), total_reward, done, self.die

    @staticmethod
    def reward_memory():
        # Record reward for the last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

    
agent = Agent('cpu')

video_dir = "/Applications/Files/SEM_7/MAJOR/RL/datavideos"  # Path where videos will be saved
os.makedirs(video_dir, exist_ok=True)  # Make sure the directory exists


env = gym.make('CarRacing-v2', verbose=1, render_mode='human', domain_randomize=False)
# env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)  # Record all episodes
env_wrap = Wrapper(env)

# Load Model
def load(agent, directory, filename):
    model_path = os.path.join(directory, filename)
    agent.net.load_state_dict(torch.load(model_path))

# Play Function
from collections import deque

def play(env, agent, n_episodes):
    state = env_wrap.reset()
    
    scores_deque = deque(maxlen=100)
    scores = []
    
    for i_episode in range(1, n_episodes + 1):
        state = env_wrap.reset()        
        score = 0
        time_start = time.time()
        
        while True:
            # Select action from the agent
            action, a_logp = agent.select_action(state)
            # plt.imshow(state[3])
            # plt.show()
            # print(state.shape)
            env.render()

            # Take a step in the wrapped environment
            next_state, reward, done, die = env_wrap.step(
                action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
            )

            state = next_state
            score += reward
            
            if done or die:
                break

        # Record time and performance
        elapsed_time = int(time.time() - time_start)
        scores_deque.append(score)
        scores.append(score)

        print(f'Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}, '
              f'\tScore: {score:.2f} \tTime: {elapsed_time // 3600:02}:{elapsed_time % 3600 // 60:02}:{elapsed_time % 60:02}')
    
    return scores


load(agent, '/Applications/Files/SEM_7/MAJOR/RL/model', 'model_weights_best.pth')

play(env, agent, n_episodes=1)
# /Users/divyansh/Downloads/model_weights_best.pth
# Close the environment
env.close()
