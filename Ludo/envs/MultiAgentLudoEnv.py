import functools
import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces
from pettingzoo.test import api_test


"""
Ludo Game Rules:
1. Each player has 4 tokens that start out of bounds.
2. Players roll a die to move their tokens clockwise around the board.
3. A 6 is required to move a token from out of bounds to the start square.
4. If a token lands on an opponent's token, the opponent's token is sent back out of bounds.
5. Tokens must reach the final square (58) exactly; overshooting aren't allowed.
6. The first player to get all 4 tokens to the final square wins.
"""


class LudoEnv(AECEnv):
    """
    A Ludo game environment implemented as an AEC (Agent Environment Cycle) environment.
    This class represents the game state, handles player actions, and manages the game flow.
    """

    # Constants
    NUM_PLAYERS = 4
    NUM_TOKENS = 4
    OUT_OF_BOUNDS = -1
    START_SQUARE = 0
    FINAL_SQUARE = 58
    DICE_MIN = 1
    DICE_MAX = 6
    CAPTURE_REWARD = 10
    MOVING_FORCE_REWARD = 1
    WIN_REWARD = 100
    ENTERING_PIECE_REWARD = 10
    SAFE_POSITION = 52
    QUARTER_RUN = 13
    render_mode = "human"

    metadata = {
        "render_modes": ["rgb_array"],
        "name": "ludo_v0",
    }

    def __init__(self) -> None:
        super().__init__()
        self.possible_agents: list[int] = list(range(self.NUM_PLAYERS))
        self.action_spaces: dict[int, spaces.Space] = self._init_action_spaces()
        self.observation_spaces: dict[int, spaces.Space] = (
            self._init_observation_spaces()
        )
        self._initialize_game_state()

    @functools.lru_cache(maxsize=NUM_PLAYERS)
    def action_space(self, agent: int) -> spaces.Discrete:
        return spaces.Discrete(self.NUM_TOKENS)

    @functools.lru_cache(maxsize=NUM_PLAYERS)
    def observation_space(self, agent: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        "board_state": spaces.Box(
                            low=self.OUT_OF_BOUNDS,
                            high=self.FINAL_SQUARE,
                            shape=(self.NUM_PLAYERS, self.NUM_TOKENS),
                            dtype=np.int8,
                        ),
                        "last_roll": spaces.Discrete(self.DICE_MAX + 1),
                    }
                )
            }
        )

    def _init_action_spaces(self) -> dict[int, spaces.Space]:
        """Create action spaces for all players."""
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    def _init_observation_spaces(self) -> dict[int, spaces.Space]:
        """Create observation spaces for all players."""
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    def _initialize_game_state(self) -> None:
        """Initialize or reset the game state and member variables."""
        self.board_state: np.ndarray = np.full(
            (self.NUM_PLAYERS, self.NUM_TOKENS), self.OUT_OF_BOUNDS, dtype=np.int8
        )
        self.dice_roll: int = 0
        self.round_count: int = 0
        self.roll_again: bool = False
        self.agent_selection: int = 0
        self.agents = self.possible_agents
        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._initialize_game_state()

    def _roll_dice(self) -> int:
        """Roll the dice and return the result."""
        return np.random.randint(self.DICE_MIN, self.DICE_MAX + 1)

    def _calculate_reward(self, action: int, new_pos: int, captured: bool) -> int:
        """Calculate the reward for a player's action."""
        return (
            self._get_move_reward(action, new_pos)
            + self._get_capture_reward(captured)
            + self._get_final_square_reward(new_pos)
            + self._get_winning_reward()
            + self._get_out_of_bounds_penalty()
        )

    def _get_move_reward(self, action: int, new_pos: int) -> int:
        """Calculate the reward for moving a token."""
        distance = int(new_pos - self.board_state[self.player_index][action])
        return distance * self.MOVING_FORCE_REWARD

    def _get_capture_reward(self, captured: bool) -> int:
        """Calculate the reward for capturing an opponent's token."""
        return self.CAPTURE_REWARD if captured else 0

    def _get_final_square_reward(self, new_pos: int) -> int:
        """Calculate the reward for reaching the final square."""
        return self.ENTERING_PIECE_REWARD if new_pos == self.FINAL_SQUARE else 0

    def _get_winning_reward(self) -> int:
        """Calculate the reward for winning the game."""
        return self.WIN_REWARD if self.is_player_done(self.player_index) else 0

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return None

        self.round_count += 1
        self.dice_roll = self._roll_dice()
        self._update_game_state(action)

    def _update_game_state(self, action: int) -> None:
        """Update the game state after a player's move."""
        new_pos = self._calculate_new_position(
            self.board_state[self.player_index][action], self.dice_roll
        )
        captured = self._check_capture(self.player_index, new_pos)
        self.board_state[self.player_index][action] = new_pos

        reward = self._calculate_reward(action, new_pos, captured)
        self.rewards[self.player_index] = reward
        self._cumulative_rewards[self.player_index] += reward

        self.roll_again = (
            self.dice_roll == self.DICE_MAX and not self.terminations[self.player_index]
        )
        if not self.roll_again:
            self.agent_selection = (self.agent_selection + 1) % self.NUM_PLAYERS

    def observe(self, player_index: int) -> dict[str, object]:
        return {
            "observation": {
                "board_state": np.roll(
                    self.board_state, -player_index * self.NUM_TOKENS
                ),
                "last_roll": self.dice_roll,
            }
        }

    def render(self) -> None:
        """Render the current game state to the console."""
        print(
            str(self.board_state).replace("\n", " "),
            self.round_count,
            f"Current player: {self.agent_selection}",
            f"Last dice roll: {self.dice_roll}",
        )

    def _calculate_new_position(self, current_pos: int, steps: int) -> int:
        """Calculate the new position of a token after moving a certain number of steps."""
        if current_pos == self.OUT_OF_BOUNDS and steps == self.DICE_MAX:
            return self.START_SQUARE
        elif current_pos + steps <= self.FINAL_SQUARE:
            return current_pos + steps
        return current_pos

    def _check_capture(self, current_player_index: int, new_position: int) -> bool:
        """Check if a capture occurs at the given position and handle it."""
        if new_position == self.START_SQUARE or new_position >= self.SAFE_POSITION:
            return False  # Starting positions and safe positions are safe

        for other_player_index in range(self.NUM_PLAYERS):
            if self._is_capture_possible(
                current_player_index, other_player_index, new_position
            ):
                return self._perform_capture(other_player_index, new_position)

        return False

    def _is_capture_possible(
        self, current_player_index: int, other_player_index: int, position: int
    ) -> bool:
        """Check if a capture is possible for the given players and position."""
        if other_player_index == current_player_index:
            return False
        is_safe_quarter_square = position % self.QUARTER_RUN == self.START_SQUARE

        capture_position = (
            position + (other_player_index - current_player_index) * self.QUARTER_RUN
        )

        return not is_safe_quarter_square and capture_position in range(
            1, self.SAFE_POSITION
        )

    def _perform_capture(self, other_player_index: int, capture_position: int) -> bool:
        """Perform the capture action and return True if a capture occurred."""
        for piece in range(self.NUM_TOKENS):
            if self.board_state[other_player_index][piece] == capture_position:
                self.board_state[other_player_index][piece] = self.OUT_OF_BOUNDS
                return True
        return False

    def is_player_done(self, player_index: int) -> bool:
        """Check if a player has finished the game."""
        return bool(np.all(self.board_state[player_index] == self.FINAL_SQUARE))

    def _get_out_of_bounds_penalty(self) -> int:
        """Small penalty for having tokens out of bounds."""
        return -1 * int(
            np.sum(self.board_state[self.player_index] == self.OUT_OF_BOUNDS)
        )

    @property
    def terminations(self) -> dict[int, bool]:
        return {
            player: self.is_player_done(player) for player in range(self.NUM_PLAYERS)
        }

    @property
    def truncations(self) -> dict[int, bool]:
        return {player: False for player in range(self.NUM_PLAYERS)}

    @property
    def infos(self) -> dict[int, dict]:
        return {player: {} for player in range(self.NUM_PLAYERS)}

    @property
    def player_index(self) -> int:
        return self.agent_selection


if __name__ == "__main__":
    env = LudoEnv()
    api_test(env)

    # env.reset(seed=42)

    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         print(f"Agent {agent} has reached the final square.")
    #         break
    #     # this is where you would insert your policy
    #     action = int(env.action_space(agent).sample())
    #     env.step(action)
    #     env.render()
    # env.close()
