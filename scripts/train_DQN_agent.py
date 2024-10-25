import os
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from pettingzoo.utils.wrappers import BaseWrapper

# Import your Ludo environment
from Ludo.envs import LudoEnv


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(BaseWrapper(LudoEnv()))


def _get_agents() -> tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()

    # Handle Dict observation space
    observation_space = env.observation_space.shape or 4 * 4 + 1
    action_space = env.action_space.shape or 4

    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")

    # Model architecture specific to Ludo
    net = Net(
        state_shape=observation_space,
        action_shape=action_space,
        hidden_sizes=[256, 256, 256],  # Larger network for more complex game
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    agent_learn: BasePolicy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.95,  # Higher discount factor for longer-term rewards
        estimation_step=5,  # Longer estimation steps for better planning
        target_update_freq=500,
        reward_normalization=True,
    )
    agent2: BasePolicy = RandomPolicy()
    agent3: BasePolicy = RandomPolicy()
    agent4: BasePolicy = RandomPolicy()

    # For Ludo, we need 4 agents (one learning, three random)
    agents = [agent_learn, agent2, agent3, agent4]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(5)])

    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(50_000, len(train_envs)),  # Larger buffer for complex game
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # Initial data collection
    train_collector.collect(n_step=1000)

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ludo", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ludo", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 2.5  # Adjust based on your reward scale

    def train_fn(epoch, env_step):
        # Decay exploration rate over time
        eps = max(0.1, 1 - epoch * 0.02)  # Linear decay from 1 to 0.1
        policy.policies[agents[0]].set_eps(eps)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)  # Small exploration during testing

    def reward_metric(rews):
        # Focus on the learning agent's rewards (first agent)
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=200,  # More epochs for complex game
        step_per_epoch=2000,  # More steps per epoch
        step_per_collect=100,
        episode_per_test=20,
        batch_size=128,  # Larger batch size
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[0]])")
