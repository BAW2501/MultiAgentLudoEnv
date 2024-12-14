import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

from Ludo.envs.MultiAgentLudoEnv import LudoEnv

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPONetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=4,  # number of tokens
        lr=0.0003,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.memory = []
        
    def _preprocess_observation(self, observation):
        board_state = observation['observation']['board_state']
        last_roll = observation['observation']['last_roll']
        
        # Flatten the board state and concatenate with last roll
        flat_board = board_state.flatten()
        input_tensor = np.concatenate([flat_board, [last_roll]])
        return torch.FloatTensor(input_tensor).to(self.device)
    
    def select_action(self, observation):
        state = self._preprocess_observation(observation)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state)
            
        # Create distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store the action details
        self.memory.append({
            'state': state,
            'action': action,
            'action_prob': action_probs[action].item(),
            'value': state_value.item()
        })
        
        return action.item()
    
    def update(self, next_observation, reward, done):
        if len(self.memory) == 0:
            return
        
        state = self.memory[-1]['state']
        action = self.memory[-1]['action']
        old_action_prob = self.memory[-1]['action_prob']
        old_value = self.memory[-1]['value']
        
        # Calculate returns and advantage
        if done:
            next_value = 0
        else:
            with torch.no_grad():
                _, next_value = self.network(self._preprocess_observation(next_observation))
                next_value = next_value.item()
        
        returns = reward + self.gamma * next_value
        advantage = returns - old_value
        
        # Get current action probabilities and value
        action_probs, value = self.network(state)
        dist = Categorical(action_probs)
        
        # Calculate ratios and surrogate losses
        new_action_prob = dist.log_prob(action).exp()
        ratio = new_action_prob / old_action_prob
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        
        # Calculate losses
        actor_loss = -torch.min(surr1, surr2)
        critic_loss = self.value_coef * (returns - value).pow(2)
        entropy_loss = -self.entropy_coef * dist.entropy()
        
        total_loss = actor_loss + critic_loss + entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.memory.clear()

def train_ppo(env, num_episodes=1000):
    # Initialize PPO agent
    input_dim = env.NUM_PLAYERS * env.NUM_TOKENS + 1  # board_state + last_roll
    agent = PPOAgent(input_dim=input_dim)
    
    episode_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            episode_reward += reward
            
            if termination or truncation:
                action = None
                agent.update(observation, reward, True)
            else:
                action = agent.select_action(observation)
                agent.update(observation, reward, False)
            
            env.step(action)
            
        episode_rewards.append(episode_reward)
        
        if episode % 1 == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent

# Usage example:
if __name__ == "__main__":
    env = LudoEnv()
    trained_agent = train_ppo(env)