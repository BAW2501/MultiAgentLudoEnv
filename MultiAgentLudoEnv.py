from enum import Enum
import numpy as np
from typing import Dict, List, Tuple, Optional
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces


class Player(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"


NUM_PLAYERS = len(Player)
NUM_TOKENS = 4
OUT_OF_BOUNDS = -1
START_SQUARE = 0
FINAL_SQUARE = 58
ILLEGAL_ACTION_PENALTY = -50


class LudoEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "ludo_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = [player.value for player in Player]

        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.possible_agents
        }  # plays one of the 4 tokens (0,1,2,3) with the dice roll (OOB token + 6 enters it )
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "board_state": spaces.Box(
                        low=OUT_OF_BOUNDS,
                        high=FINAL_SQUARE,
                        shape=(NUM_PLAYERS, NUM_TOKENS),
                        dtype=np.int8,
                    ),
                    "current_player": spaces.Discrete(NUM_PLAYERS),
                    "action_mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                    "last_roll": spaces.Discrete(7),
                }
            )
            for agent in self.possible_agents
        }

        self.state: np.ndarray = np.full(
            (NUM_PLAYERS, NUM_TOKENS), OUT_OF_BOUNDS, dtype=np.int8
        )
        self.current_player: str = Player.RED.value
        self.dice_roll: int = 0
        self.agent_selection: str = self.current_player
        self.round_count: int = 0

        self.start_positions: List[int] = [0, 13, 26, 39]
        self.home_stretches: List[List[int]] = [
            list(range(51, 57)),
            list(range(12, 18)),
            list(range(25, 31)),
            list(range(38, 44)),
        ]

        self.roll_again: bool = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        self.state = np.full((NUM_PLAYERS, NUM_TOKENS), OUT_OF_BOUNDS, dtype=np.int8)
        self.agents = self.possible_agents[:]
        self.current_player = Player.RED.value
        self.dice_roll = 0
        self.round_count = 0
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.agent_selection = self.current_player
        self._agent_selector = agent_selector(self.agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.roll_again = False

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.round_count += 1
        agent = self.agent_selection
        player_index = self.possible_agents.index(agent)

        # Roll the dice at the beginning of each turn
        self.dice_roll = np.random.randint(1, 7)

        # Check if the action is legal
        if not self._is_action_legal(action):
            self.rewards[agent] = ILLEGAL_ACTION_PENALTY
            self._cumulative_rewards[agent] += ILLEGAL_ACTION_PENALTY
            self.roll_again = False
        else:
            reward = 0

            if action == 0 and self.dice_roll == 6:  # Enter a token
                for token in range(NUM_TOKENS):
                    if self.state[player_index][token] == OUT_OF_BOUNDS:
                        self.state[player_index][token] = self.start_positions[
                            player_index
                        ]
                        break
            elif 1 <= action <= 4:  # Move a token
                token = action - 1
                if self.state[player_index][token] != OUT_OF_BOUNDS:
                    new_pos = self._calculate_new_position(
                        self.state[player_index][token], self.dice_roll
                    )
                    self.state[player_index][token] = new_pos

                    captured = self._check_capture(player_index, new_pos)
                    if captured:
                        reward += 10

                    if new_pos == FINAL_SQUARE:
                        reward += 100

            self.rewards[agent] = reward
            self._cumulative_rewards[agent] += reward

            # Set roll_again flag if dice_roll is 6
            self.roll_again = self.dice_roll == 6
        self.terminations[player_index] = self.get_playerDone(player_index) 

        if self._check_game_over():
            self.terminations = {agent: True for agent in self.agents}
        elif not self.roll_again:
            # Only change the agent if we're not rolling again
            self.agent_selection = self._agent_selector.next()

    def _is_action_legal(self, action: int) -> bool:
        mask = self._mask_actions(self.agent_selection)
        return mask[action] == 1

    def observe(self, agent: str) -> Dict:
        return {
            "board_state": self.state,
            "current_player": self.possible_agents.index(self.agent_selection),
            "action_mask": self._mask_actions(agent),
            "last_roll": self.dice_roll,
        }

    def render(self) -> None:
        print(
            str(self.state).replace("\n", " "),
            self.round_count,
            # f"Current player: {self.agent_selection}",
            # f"Last dice roll: {self.dice_roll}",
            end="\r",
        )

    def _calculate_new_position(self, current_pos: int, steps: int) -> int:
        if current_pos == OUT_OF_BOUNDS and steps == 6:
            return START_SQUARE
        elif current_pos in range(52):
            return current_pos + steps
        elif current_pos + steps <= FINAL_SQUARE:
            return current_pos + steps
        return current_pos

    def _check_capture(
        self, current_player_index: int, position: int
    ) -> bool:
        if position == START_SQUARE or position >= 52:
            return False  # Starting positions are safe

        for player in range(NUM_PLAYERS):
            if player != current_player_index:
                for piece in range(NUM_TOKENS):
                    # each player starts 13 spaces away from each other
                    if self.state[player][piece] == position + (player - current_player_index) * 13:
                        # Capture occurred
                        self.state[player][piece] = OUT_OF_BOUNDS
                        return True

        return False  # No capture occurred

    def _mask_actions(self, agent: str) -> np.ndarray:
        mask = np.zeros(4, dtype=np.int8)
        player_index = self.possible_agents.index(agent)

        # if player has any out of bounds pieces and has rolled a 6 then action is allowed
        if self.dice_roll == 6:
            mask = self.state[player_index] == OUT_OF_BOUNDS

        # if player has a piece inside the board and their dice roll doesn't overshoot final square then action is allowed
        for token in range(NUM_TOKENS):
            if (
                START_SQUARE
                <= self.state[player_index][token]
                <= FINAL_SQUARE - self.dice_roll
            ):
                mask[token] = 1 # moving pieces that aren't out of bounds
            if self.state[player_index][token] == FINAL_SQUARE:
                mask[token] = 0

        return mask

    def _check_game_over(self) -> bool:
        return any(
            np.all(player_pieces == FINAL_SQUARE) for player_pieces in self.state
        )
    def get_playerDone(self,player_index):
        return bool(np.all(self.state[player_index] == FINAL_SQUARE))
        
    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    env = LudoEnv()
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break
        # this is where you would insert your policy
        action = env.action_space(agent).sample()  
        env.step(action)
        env.render()
    env.close()
