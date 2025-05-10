import torch
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class TrackFeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2)  # Changed input channels to 4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (B, 4, 96, 96) → (B, 16, 48, 48)
        x = F.relu(self.conv2(x))  # (B, 32, 24, 24)
        x = F.relu(self.conv3(x))  # (B, 64, 12, 12)
        x = F.relu(self.conv4(x))  # (B, 64, 6, 6)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



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


# Load trained model
model = torch.load('expert_runs/model_full.pth')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up environment
env = gym.make("CarRacing-v2", render_mode="human", domain_randomize=True)
obs, _ = env.reset()
done = False

# Init stack of 4 binary masks (grayscale 96x96)
frame_stack = deque(maxlen=4)
center_color = [None, None, None]
frame_count = 0

# Fill with initial identical masks
initial_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # normalize
for _ in range(4):
    frame_stack.append(initial_mask)


while not done:
    # Stack into (4, 96, 96), normalize already done
    stack_np = np.stack(frame_stack, axis=0)
    input_tensor = torch.tensor(stack_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 96, 96)

    with torch.no_grad():
        action = model(input_tensor).squeeze().cpu().numpy()

    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1] = np.clip(action[1], 0.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)

    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    frame_count += 1
    print(frame_count)
    new_mask = get_binary_mask(obs, frame_count, center_color) / 255.0
    frame_stack.append(new_mask)

env.close()



# import matplotlib.pyplot as plt

# # Set up environment
# env = gym.make("CarRacing-v2", render_mode="human", domain_randomize=True)
# obs, _ = env.reset()
# done = False

# # Init stack of 4 binary masks (grayscale 96x96)
# frame_stack = deque(maxlen=4)
# center_color = [None, None, None]
# frame_count = 0

# # Fill with initial identical masks
# initial_mask = get_binary_mask(obs, frame_count, center_color) / 255.0  # normalize
# for _ in range(4):
#     frame_stack.append(initial_mask)

# # Set up the plot
# plt.ion()  # Turn on interactive mode for live plotting
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a subplot with 2 columns

# while not done:
#     # Stack into (4, 96, 96), normalize already done
#     stack_np = np.stack(frame_stack, axis=0)
#     input_tensor = torch.tensor(stack_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 96, 96)

#     with torch.no_grad():
#         action = model(input_tensor).squeeze().cpu().numpy()

#     action[0] = np.clip(action[0], -1.0, 1.0)
#     action[1] = np.clip(action[1], 0.0, 1.0)
#     action[2] = np.clip(action[2], 0.0, 1.0)

#     obs, _, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated

#     frame_count += 1
#     new_mask = get_binary_mask(obs, frame_count, center_color) / 255.0
#     frame_stack.append(new_mask)

#     # Plot the current RGB image and binary mask
#     # Plot RGB Image
#     axes[0].clear()
#     axes[0].imshow(obs)
#     axes[0].set_title("RGB Image")
#     axes[0].axis('off')

#     # Plot Binary Mask
#     axes[1].clear()
#     axes[1].imshow(new_mask, cmap='gray')
#     axes[1].set_title("Binary Mask")
#     axes[1].axis('off')

#     plt.draw()
#     plt.pause(0.01)  # Pause to update the plot

# env.close()

# plt.ioff()  # Turn off interactive mode after the loop ends
# plt.show()  # Keep the final plot displayed
