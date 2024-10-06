from gymnasium.envs.registration import register
from .envs import MultiAgentLudoEnv

# Register the multi-agent Ludo environment
register(
    id="MultiAgentLudo-v0",
    entry_point="Ludo.envs:MultiAgentLudoEnv",
)

# You can add more versions or variations here as needed

# Import the environments to make them accessible when importing the package
from .envs import MultiAgentLudoEnv

__all__ = ["MultiAgentLudoEnv"]