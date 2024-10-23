# from calendar import c
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions.categorical import Categorical
# import numpy as np

# from Ludo.envs import LudoEnv

# class LudoAgent(nn.Module):
#     def __init__(self, observation_space, action_space):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(observation_space, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU()
#         )
#         self.actor = nn.Linear(64, action_space)
#         self.critic = nn.Linear(64, 1)

#     def get_value(self, x):
#         return self.critic(self.network(x))

#     def get_action_and_value(self, x, action=None):
#         hidden = self.network(x)
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer



# def train_ppo(env, num_episodes, batch_size, learning_rate, gamma, clip_coef, max_steps):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize the environment
#     env = LudoEnv()
#     num_agents = env.NUM_PLAYERS
#     observation_space = env.observation_spaces[0]["board_state"].shape[0] * num_agents + 1  # board state + last roll
#     action_space = env.action_spaces[0].n

#     # Initialize the agent
#     agent = LudoAgent(observation_space, action_space).to(device)
#     optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

#     for episode in range(num_episodes):
#         done = False
#         episode_reward = 0
#         episode_length = 0

#         observations = []
#         actions = []
#         logprobs = []
#         rewards = []
#         values = []

#         while not done and episode_length < max_steps:
#             current_player = env.agent_selection
#             obs = env.observe(current_player)
#             current_obs = torch.FloatTensor(
#                 np.concatenate([obs["board_state"].flatten(), 
#                                 [obs["last_roll"]]])
#             ).to(device)
#             print(current_obs.shape)

#             with torch.no_grad():
#                 action, logprob, _, value = agent.get_action_and_value(current_obs)

#             observations.append(current_obs)
#             actions.append(action)
#             logprobs.append(logprob)
#             values.append(value)

#             obs, reward, done, _ = env.step(action.item())
#             rewards.append(reward[current_player])
#             episode_reward += reward[current_player]
#             episode_length += 1

#         # Compute returns and advantages
#         returns = compute_gae(rewards, values, gamma)
#         advantages = returns - torch.cat(values).squeeze()

#         # PPO update
#         update_ppo(agent, optimizer, observations, actions, logprobs, returns, advantages, clip_coef)

#         print(f"Episode {episode + 1}, Reward: {episode_reward}, Length: {episode_length}")

# def compute_gae(rewards, values, gamma, lambda_=0.95):
#     gae = 0
#     returns = []
#     for step in reversed(range(len(rewards))):
#         delta = rewards[step] + gamma * values[step + 1] - values[step]
#         gae = delta + gamma * lambda_ * gae
#         returns.insert(0, gae + values[step])
#     return torch.tensor(returns)

# def update_ppo(agent, optimizer, observations, actions, old_logprobs, returns, advantages, clip_coef):
#     observations = torch.cat(observations)
#     actions = torch.cat(actions)
#     old_logprobs = torch.cat(old_logprobs)
#     returns = returns.detach()
#     advantages = advantages.detach()

#     for _ in range(10):  # Number of optimization epochs
#         _, new_logprobs, entropy, new_values = agent.get_action_and_value(observations, actions)
        
#         ratio = (new_logprobs - old_logprobs).exp()
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages
#         actor_loss = -torch.min(surr1, surr2).mean()
        
#         critic_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
#         loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# # Run the training
# env = LudoEnv()
# train_ppo(env, num_episodes=1000, batch_size=64, learning_rate=3e-4, gamma=0.99, clip_coef=0.2, max_steps=100)