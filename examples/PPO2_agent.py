import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from Ludo.envs.MultiAgentLudoEnv import LudoEnv

# Neural Network for PPO
class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.board_size = obs_space["observation"]["board_state"].shape[0] * obs_space["observation"]["board_state"].shape[1]
        self.fc1 = nn.Linear(self.board_size + 1, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, action_space.n)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


# PPO Agent
class PPOAgent:
    def __init__(self, obs_space, action_space, gamma=0.99, lam=0.95, clip_ratio=0.2, lr=0.0003, num_epochs=4, mini_batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(obs_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        
    def select_action(self, state):
        state_tensor = self._process_state(state).to(self.device)
        probs, _ = self.actor_critic(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), probs

    def _process_state(self, state):
        board_state = state["observation"]["board_state"].flatten()
        last_roll = np.array([state["observation"]["last_roll"]])
        state_combined = np.concatenate((board_state, last_roll))
        return torch.tensor(state_combined, dtype=torch.float32).unsqueeze(0)

    def calculate_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self, states, actions, log_probs, rewards, dones):
        values = []
        for state in states:
            state_tensor = self._process_state(state).to(self.device)
            _, value = self.actor_critic(state_tensor)
            values.append(value.item())
        values.append(0) 
        
        advantages = self.calculate_advantages(rewards, values, dones).to(self.device)
        
        values = torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        returns = advantages + values


        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        
        states = torch.cat([self._process_state(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)

        for _ in range(self.num_epochs):
            for i in range(0, len(states), self.mini_batch_size):
                batch_states = states[i:i+self.mini_batch_size]
                batch_actions = actions[i:i+self.mini_batch_size]
                batch_old_log_probs = old_log_probs[i:i+self.mini_batch_size]
                batch_advantages = advantages[i:i+self.mini_batch_size]
                batch_returns = returns[i:i+self.mini_batch_size]

                probs, values = self.actor_critic(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = (batch_returns - values).pow(2).mean()
                
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Training Loop
def train_ppo(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        env.reset(seed=42)
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        total_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                #action = None
                dones.append(True)
            else:
              
                action, log_prob, _ = agent.select_action(observation)
                states.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                dones.append(False)
                

                
            env.step(action)

            
            total_rewards[agent_id] += reward
            
            if env.rewards[agent_id] is not None :
               rewards.append(env.rewards[agent_id])
        if len(states) > 0 :

          agent.update(states, actions, log_probs, rewards, dones)

        print(f"Episode: {episode+1}, Rewards: {total_rewards}")


# Main execution
if __name__ == "__main__":
    env = LudoEnv()
    
    agent = PPOAgent(env.observation_space(0), env.action_space(0))
    train_ppo(env, agent, num_episodes=2000)