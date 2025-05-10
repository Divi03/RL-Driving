import gymnasium as gym
import torch
import numpy as np
from collections import deque
from torch.distributions import Beta
import torch.nn.functional as F
from torch import nn
import cv2

import torch.optim as optim
from torch.distributions import Beta
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# --- Masking Pipeline ---
def apply_gaussian_blur(channel, ksize=5):
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)

def adjust_brightness(channel, factor=1.2):
    channel = np.clip(np.float32(channel) * factor, 0, 255)
    return np.uint8(channel)

def adjust_saturation_single_channel(channel, factor=1.5):
    rgb = cv2.merge([channel, channel, channel])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return cv2.cvtColor(saturated, cv2.COLOR_RGB2GRAY)

def extract_center_value(channel, region_size=7, offset_y=10):
    h, w = channel.shape
    cx, cy = w // 2, h // 2
    offsets = [-offset_y, 0, offset_y]
    values = []
    for off in offsets:
        y1 = cy + off - region_size // 2
        y2 = cy + off + region_size // 2
        x1 = cx - region_size // 2
        x2 = cx + region_size // 2
        region = channel[y1:y2, x1:x2]
        values.append(np.mean(region))
    return np.mean(values)

def extract_mask(channel, center_value, tolerance=25):
    diff = np.abs(channel.astype(np.int16) - int(center_value))
    mask = np.uint8(diff < tolerance) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def mask_quality(mask):
    white = np.sum(mask == 255)
    total = mask.size
    return white / total

def combine_masks(mask_r, mask_g, mask_b, threshold=127):
    qualities = [mask_quality(mask_r), mask_quality(mask_g), mask_quality(mask_b)]
    valid = [(0.05 < q < 0.29) for q in qualities]
    masks = [mask_r, mask_g, mask_b]
    used = [m for m, v in zip(masks, valid) if v]
    if not used:
        return np.zeros_like(mask_r)
    weight = 1.0 / len(used)
    combined = sum(weight * m for m in used)
    _, binary = cv2.threshold(combined.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    return binary

def get_binary_mask(obs, frame_count, center_color):
    r, g, b = cv2.split(obs)
    r_proc = adjust_saturation_single_channel(adjust_brightness(apply_gaussian_blur(r, 5), factor=0.9), factor=1.5)
    g_proc = adjust_saturation_single_channel(adjust_brightness(apply_gaussian_blur(g, 5), factor=0.9), factor=1.5)
    b_proc = adjust_saturation_single_channel(adjust_brightness(apply_gaussian_blur(b, 5), factor=0.9), factor=1.5)

    if frame_count == 15:
        r_val = extract_center_value(r)
        g_val = extract_center_value(g)
        b_val = extract_center_value(b)
        center_color[:] = [r_val, g_val, b_val]

    if center_color[0] is not None:
        mask_r = extract_mask(r_proc, center_color[0])
        mask_g = extract_mask(g_proc, center_color[1])
        mask_b = extract_mask(b_proc, center_color[2])
        mask = combine_masks(mask_r, mask_g, mask_b)
        # print(f"Mask shape: {mask.shape}, Mask quality: {mask_quality(mask)}")  # Debugging the mask
        return mask
    else:
        return np.zeros_like(r)


# Simplified PPO Network
class Net(nn.Module):
    """
    Convolutional Neural Network for PPO with a simplified architecture
    """

    def __init__(self, img_stack):
        super(Net, self).__init__()

        # Convolutional layers with simplified architecture
        self.conv1 = nn.Conv2d(img_stack, 16, kernel_size=5, stride=2, padding=2)  # Changed input channels to 4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.v = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # Heads for action distribution (alpha, beta)
        self.alpha_head = nn.Sequential(nn.Linear(128, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(128, 3), nn.Softplus())

        # Initialize weights
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flattening for fully connected layers
        x = x.view(-1, 64 * 6 * 6)

        # Fully connected layers
        x = F.relu(self.fc1(x))

        # Value function (V)
        v = self.v(x)

        # Action distribution heads (alpha and beta for Beta distribution)
        alpha = self.alpha_head(x) + 1  # Ensure positivity of alpha
        beta = self.beta_head(x) + 1    # Ensure positivity of beta

        return (alpha, beta), v


GAMMA=0.99
EPOCH= 8 # beter than 10
MAX_SIZE = 2000 ## CUDA out of mem for max_size=10000
BATCH=128 
EPS=0.1
LEARNING_RATE = 0.001 # bettr than 0.005 or 0.002 

# PPO Agent Class
# Ensure transition is a structured numpy array
transition = np.dtype([('s', np.float64, (4, 96, 96)), 
                       ('a', np.float64, (3,)), 
                       ('a_logp', np.float64),
                       ('r', np.float64), 
                       ('s_', np.float64, (4, 96, 96))])

class Agent:
    def __init__(self, device):
        self.training_step = 0
        self.net = Net(img_stack=4).double().to(device)
        self.buffer = np.empty(MAX_SIZE, dtype=transition)  # Define buffer with dtype as transition
        self.counter = 0
        self.device = device
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def store(self, transition):
        """ Store transition in buffer. Ensure transition matches dtype. """
        # Check if transition matches the dtype
        transition_struct = (
            transition['s'], 
            transition['a'], 
            transition['a_logp'], 
            transition['r'], 
            transition['s_']
        )
        self.buffer[self.counter] = transition_struct
        self.counter += 1
        if self.counter == MAX_SIZE:
            self.counter = 0
            return True
        else:
            return False

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

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        next_s = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + GAMMA * self.net(next_s)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(EPOCH):
            for index in BatchSampler(SubsetRandomSampler(range(MAX_SIZE)), BATCH, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                # Ensure actions are within the valid range for Beta distribution
                a_normalized = a[index].clamp(min=1e-5, max=1-1e-5)  # Avoid issues with log(0) by clamping
                a_logp = dist.log_prob(a_normalized).sum(dim=1, keepdim=True)

                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print action loss per epoch
                if self.training_step % 100 == 0:
                    print(f"Epoch {self.training_step}: Action Loss: {action_loss.item()}")

    # Add a method to save the model's state
    def save_model(self, filename):
        torch.save(self.net.state_dict(), filename)
        print(f"Model saved to {filename}")



# def run_ppo_model():
#     # Set up the device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Initialize PPO model
#     model = Net(img_stack=4)
#     model.eval()  # Set the model to evaluation mode
#     model = model.to(device)

#     # Load the saved model (replace with your model file path)
#     model_path = "/Users/divyansh/Downloads/model_ppo_w_r.pth"  # Path to your saved model
#     model.load_state_dict(torch.load(model_path))  # Load model state dict
#     print(f"Model loaded from {model_path}")

#     # Set up the environment
#     env = gym.make("CarRacing-v2", render_mode='human',domain_randomize=True)
#     obs, _ = env.reset()
#     done = False

#     # Initialize frame stack (4 binary masks of size 96x96)
#     frame_stack = deque(maxlen=4)
#     center_color = [None, None, None]  # Example, adjust if you track a specific color
#     frame_count = 0

#     # Fill stack with initial identical masks
#     initial_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # Normalize mask to [0, 1]
#     for _ in range(4):
#         frame_stack.append(initial_mask)

#     total_reward = 0  # Initialize total reward

#     while not done:
#         # Stack into (4, 96, 96), normalize already done
#         stack_np = np.stack(frame_stack, axis=0)
#         input_tensor = torch.tensor(stack_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 96, 96)

#         # Get action from model
#         with torch.no_grad():
#             (alpha, beta), _ = model(input_tensor)
#         dist = Beta(alpha, beta)
#         action = dist.sample()
#         action = action[0]
#         action[0] = np.clip(action[0], -1.0, 1.0)  # Steering
#         action[1] = np.clip(action[1], 0.0, 1.0)  # Gas
#         action[2] = np.clip(action[2], 0.0, 1.0)  # Brake

#         # Step in the environment with the selected action
#         obs, reward, terminated, truncated, _ = env.step(action.squeeze().cpu().numpy())
#         done = terminated or truncated

#         # Accumulate the reward
#         total_reward += reward

#         # Update the frame stack
#         frame_count += 1
#         new_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # Normalize
#         frame_stack.append(new_mask)

#     env.close()

#     # Print final reward
#     print(f"Final reward: {total_reward}")


# Run the model (for testing the trained agent)
# Run the model (for testing the trained agent)
def run_ppo_model():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize PPO model
    model = Net(img_stack=4)
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)

    # Load the saved model (replace with your model file path)
    model_path = "/Users/divyansh/Downloads/model_ppo_w_r.pth"  # Path to your saved model
    model.load_state_dict(torch.load(model_path))  # Load model state dict
    print(f"Model loaded from {model_path}")

    # Set up the environment
    env = gym.make("CarRacing-v2", render_mode='human' ,domain_randomize=False)  # Disable domain randomization for easier learning
    obs, _ = env.reset()
    done = False

    # Initialize frame stack (4 binary masks of size 96x96)
    frame_stack = deque(maxlen=4)
    center_color = [None, None, None]  # Example, adjust if you track a specific color
    frame_count = 0

    # Fill stack with initial identical masks
    initial_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # Normalize mask to [0, 1]
    for _ in range(4):
        frame_stack.append(initial_mask)

    total_reward = 0  # Initialize total reward

    while not done:
        # Stack into (4, 96, 96), normalize already done
        stack_np = np.stack(frame_stack, axis=0)
        input_tensor = torch.tensor(stack_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 96, 96)

        # Get action from model
        with torch.no_grad():
            (alpha, beta), _ = model(input_tensor)
        dist = Beta(alpha, beta)
        action = dist.sample()
        action = action[0]
        action[0] = np.clip(action[0], -1.0, 1.0)  # Steering
        action[1] = np.clip(action[1], 0.0, 1.0)  # Gas
        action[2] = np.clip(action[2], 0.0, 1.0)  # Brake

        # Step in the environment with the selected action
        obs, reward, terminated, truncated, _ = env.step(action.squeeze().cpu().numpy())
        done = terminated or truncated

        # Accumulate the reward
        total_reward += reward

        # Update the frame stack
        frame_count += 1
        new_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # Normalize
        frame_stack.append(new_mask)

    env.close()

    # Print final reward
    print(f"Final reward: {total_reward}")

# Run the model
if __name__ == "__main__":
    run_ppo_model()
