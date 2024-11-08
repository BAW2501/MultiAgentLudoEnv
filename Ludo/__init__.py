from gymnasium.envs.registration import register
from .envs import MultiAgentLudoEnv

# Register the multi-agent Ludo environment
register(
    id="MultiAgentLudo-v0",
    entry_point="Ludo.envs:MultiAgentLudoEnv",
)

__all__ = ["MultiAgentLudoEnv"]
