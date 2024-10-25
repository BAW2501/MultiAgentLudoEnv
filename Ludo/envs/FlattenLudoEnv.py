from gymnasium import spaces
import numpy as np
from .MultiAgentLudoEnv import LudoEnv


class FlatLudoEnv(LudoEnv):
    """
    A wrapper for the Ludo environment that flattens the observation space.
    Inherits from AECEnv to maintain PettingZoo compatibility.
    """

    def __init__(self):
        super().__init__()

        flat_dim = self.NUM_PLAYERS * self.NUM_TOKENS + 1

        # Create new flattened observation spaces
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1,  # Minimum value (-1 for OUT_OF_BOUNDS)
                high=max(self.FINAL_SQUARE, self.DICE_MAX),  # Maximum value
                shape=(flat_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def _flatten_observation(self, obs):
        """Convert the dictionary observation to a flat array."""
        board_state = obs["board_state"].flatten()
        last_roll = np.array([obs["last_roll"]])
        return np.concatenate([board_state, last_roll]).astype(np.float32)

    def observe(self, agent):
        """Get the observation for an agent, flattened."""
        obs = super().observe(agent)
        return self._flatten_observation(obs)
