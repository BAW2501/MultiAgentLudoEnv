from gymnasium.envs.registration import register
from .envs import MultiAgentLudoEnv
from .envs import FlatLudoEnv

# Register the multi-agent Ludo environment
register(
    id="MultiAgentLudo-v0",
    entry_point="Ludo.envs:MultiAgentLudoEnv",
)
register(
    id="FlattenLudo-v0",
    entry_point="Ludo.envs:FlattenLudoEnv",
)



__all__ = ["MultiAgentLudoEnv", "FlatLudoEnv"]