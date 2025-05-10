<!-- 
# Project Ideas on Reinforcement Learning and Supervised Imitation Learning

## Idea 1: Supervised Learning (SL) in Imitation Learning
- [ ] **Objective**: Train an agent to mimic expert behavior using SL.
- **Approach**:
  - [ ] Collect expert demonstrations in the OpenAI Gym car racing environment.
  - [ ] Use these demonstrations to train a supervised model (e.g., neural network) to predict actions based on states.
  - [ ] Evaluate the model’s performance against the expert.

## Idea 2: Reinforcement Learning (RL) for Autonomous Control
- [ ] **Objective**: Develop an RL agent to learn optimal policies through exploration and exploitation.
- **Approach**:
  - [ ] Implement an RL algorithm (e.g., DDPG, PPO) in the car racing environment.
  - [ ] Reward the agent for desirable actions (e.g., staying on track, avoiding collisions).
  - [ ] Incorporate mechanisms to handle mistakes (e.g., adaptive learning rate, replay buffer).

## Idea 3: Combining SL and RL for Enhanced Performance
- [ ] **Objective**: Leverage SL data to improve RL training efficiency and performance.
- **Approach**:
  - [ ] Use SL data to initialize the RL agent’s policy.
  - [ ] Employ batch normalization during RL training using statistics derived from SL data, ensuring they are adapted for RL contexts.
  - [ ] Analyze performance improvements by comparing RL agent’s performance with and without SL data integration.

## Summary
- Each idea builds upon the previous one, allowing for a comparative analysis of learning methods.
- The integration of SL data into RL could offer insights into how imitation learning can enhance the robustness of RL agents. -->


# 🚘 AI Navigation in OpenAI Gym using Reinforcement Learning

This repository implements a self-driving car agent using reinforcement learning techniques to navigate OpenAI Gym's CarRacing-v2 environment. The goal is to keep the car within track limits while achieving high performance across randomly generated tracks.

---

## 🎥 Demo Video

Watch the demo of the self-driving car agent in action:

[![Watch the video](https://stable-baselines3.readthedocs.io/en/master/_static/logo.png)](https://www.youtube.com/watch?v=rFwQDDbYTm4)

---

## 🧪 How to Run

- **DQN Agent**  
  `run/play_policy_template.py`  
  ➤ Runs the initial DQN-based discrete control policy.

- **PPO Agent**  
  `RL/result/Agent_play.py`  
  ➤ Runs the trained PPO agent with segmentation and reward shaping.

- **Knowledge Distillation Agent**  
  `random_domain_agent.py`  
  ➤ Runs the compressed student agent trained via knowledge distillation.

---

## 📦 Trained Models

🔗 [Download All Models from Google Drive](https://drive.google.com/drive/folders/1C7fx1pMZig1eAmuwkWHhjivr-_Ou2DSK?usp=sharing)

Includes:
- DQN Model  
- PPO Model  
- KD Student Model  
---