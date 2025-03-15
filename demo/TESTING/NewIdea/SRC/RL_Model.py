class RLAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.model(state_tensor)
            return torch.argmax(action_probs, dim=1).item()

def train_rl_agent(env_name="CartPole-v1", model, num_episodes=100):
    env = gym.make(env_name)
    agent = RLAgent(model)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)  # Use imitation model to select action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    env.close()

# Train the RL agent with the imitation model
train_rl_agent(model=imitation_model)
