import numpy as np
from typing import Optional
from pettingzoo import AECEnv
from gymnasium import spaces


NUM_PLAYERS = 4
NUM_TOKENS = 4
OUT_OF_BOUNDS = -1
START_SQUARE = 0
FINAL_SQUARE = 58


class LudoEnv(AECEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "ludo_v0"}
    CAPTURE_REWARD = 10
    MOVING_FORCE_REWARD = 1
    WIN_REWARD = 100  # SHOULD SCALE WITH THE WIN ORDER LAST ONE GETS ZERO
    ENTERING_PIECE_REWARD = 10

    def __init__(self):
        super().__init__()
        self.possible_agents = list(range(NUM_PLAYERS))

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
                    "last_roll": spaces.Discrete(7),
                }
            )
            for agent in self.possible_agents
        }

        self.state: np.ndarray = np.full(
            (NUM_PLAYERS, NUM_TOKENS), OUT_OF_BOUNDS, dtype=np.int8
        )
        self.dice_roll: int = 0
        self.round_count: int = 0
        self.roll_again: bool = False
        self.agent_selection: int = 0
        # member variables
        self.agents = self.possible_agents
        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}
        self.infos = {i: {} for i in self.possible_agents}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.state: np.ndarray = np.full(
            (NUM_PLAYERS, NUM_TOKENS), OUT_OF_BOUNDS, dtype=np.int8
        )
        self.dice_roll: int = 0
        self.round_count: int = 0
        self.roll_again: bool = False
        self.agent_selection: int = 0
        # member variables
        self.agents = self.possible_agents
        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}
        self.infos = {i: {} for i in self.possible_agents}

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.round_count += 1
        player_index = self.agent_selection

        # Roll the dice at the beginning of each turn
        self.dice_roll = np.random.randint(1, 7)

        new_pos = self._calculate_new_position(
            self.state[player_index][action], self.dice_roll
        )
        # check if piece is captured
        captured = self._check_capture(player_index, new_pos)
        # reward for how much the piece moves
        # should reward entering a piece more than moving
        distance = new_pos - self.state[player_index][action]
        # check if piece is in final square
        in_final = self.state[player_index][action] == FINAL_SQUARE

        self.state[player_index][action] = new_pos

        reward = sum(
            [
                captured * self.CAPTURE_REWARD,
                distance * self.MOVING_FORCE_REWARD,
                in_final * self.ENTERING_PIECE_REWARD,
            ]
        )

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        # Set roll_again flag if dice_roll is 6
        self.roll_again = self.dice_roll == 6 and not self.terminations[player_index]

        if not self.roll_again:
            # Only change the agent if we're not rolling again
            self.agent_selection = (self.agent_selection + 1) % NUM_PLAYERS

    def observe(self, agent: int) -> dict:
        return {
            "board_state": self.state,
            "current_player": self.possible_agents.index(self.agent_selection),
            "last_roll": self.dice_roll,
        }

    def render(self) -> None:
        print(
            str(self.state).replace("\n", " "),
            self.round_count,
            f"Current player: {self.agent_selection}",
            f"Last dice roll: {self.dice_roll}",
            # end="\r",
        )

    def _calculate_new_position(self, current_pos: int, steps: int) -> int:
        if current_pos == OUT_OF_BOUNDS and steps == 6:
            return START_SQUARE
        elif current_pos + steps <= FINAL_SQUARE:
            return current_pos + steps
        return current_pos

    def _check_capture(self, current_player_index: int, position: int) -> bool:
        if position == START_SQUARE or position >= 52:
            return False  # Starting positions are safe

        for other_player_index in range(NUM_PLAYERS):
            capture_position = (
                position + (other_player_index - current_player_index) * 13
            )
            if (
                other_player_index != current_player_index
                and capture_position in range(1, 52)
            ):
                for piece in range(NUM_TOKENS):
                    if self.state[other_player_index][piece] == capture_position:
                        # Capture occurred
                        self.state[other_player_index][piece] = OUT_OF_BOUNDS
                        return True

        return False  # No capture occurred

    def get_playerDone(self, player_index):
        return bool(np.all(self.state[player_index] == FINAL_SQUARE))

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def terminations(self) -> dict[int, bool]:
        return {player: self.get_playerDone(player) for player in range(NUM_PLAYERS)}

    @property
    def truncations(self) -> dict[int, bool]:
        return {player: False for player in range(NUM_PLAYERS)}


if __name__ == "__main__":
    env = LudoEnv()
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} has reached the final square.")
            break
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        env.step(action)
        env.render()
    env.close()
