import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAgentLudoEnv(gym.Env):
    def __init__(self):
        super(MultiAgentLudoEnv, self).__init__()
        
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Dict({
            'board_state': spaces.Box(low=-1, high=58, shape=(4, 4), dtype=int),
            'current_player': spaces.Discrete(4),
            'last_roll': spaces.Discrete(7)
        })
        
        self.state = [[-1, -1, -1, -1] for _ in range(4)]
        self.current_player = 0
        self.dice_roll = 0
        
        self.start_positions = [0, 13, 26, 39]
        self.home_stretches = [
            list(range(51, 57)),
            list(range(12, 18)),
            list(range(25, 31)),
            list(range(38, 44))
        ]
        
    def reset(self, options = None,seed=None):
        super().reset(seed=seed)
        self.state = np.full((4, 4), -1, dtype=int)
        self.current_player = 0
        self.dice_roll = 0
        return self._get_obs(), self._get_info()
    def step(self, action_dict):
        assert len(action_dict) == 4, "Must provide actions for all 4 players"
        
        rewards = {i: 0 for i in range(4)}
        dones = {i: False for i in range(4)}
        
        self.dice_roll = np.random.randint(1, 7)
        
        action = action_dict[self.current_player]
        
        if action < 4:  # Move a piece
            piece = action
            current_pos = self.state[self.current_player][piece]
            
            if current_pos == -1 and self.dice_roll == 6:
                # Move piece onto the board
                self.state[self.current_player][piece] = self.start_positions[self.current_player]
                rewards[self.current_player] = 1
            elif current_pos >= 0:
                # Move piece on the board
                new_pos = self._calculate_new_position(current_pos, self.dice_roll)
                if new_pos < 58:
                    # Check for capture
                    captured = self._check_capture(new_pos)
                    if captured:
                        rewards[self.current_player] += 2  # Reward for capturing
                        rewards[captured[0]] -= 1  # Penalty for being captured
                    
                    self.state[self.current_player][piece] = new_pos
                    rewards[self.current_player] += 0.1
                    if new_pos == 57:  # Piece reached home
                        rewards[self.current_player] += 5

        # Check for winning condition
        if np.all(self.state[self.current_player] == 58):
            dones[self.current_player] = True
            rewards[self.current_player] = 100
            dones['__all__'] = True
        
        # Move to next player if didn't roll a 6
        if self.dice_roll != 6:
            self.current_player = (self.current_player + 1) % 4
        
        return self._get_obs(), rewards, dones, False, self._get_info()
    
    def _calculate_new_position(self, current_pos, steps):
        if current_pos < 52:  # On the main track
            new_pos = (current_pos + steps) % 52
            if new_pos in self.home_stretches[self.current_player]:
                # Enter home stretch
                return 52 + (new_pos - self.home_stretches[self.current_player][0])
            return new_pos
        elif current_pos < 57:  # In the home stretch
            new_pos = current_pos + steps
            return min(new_pos, 58)  # Cap at 58 (finished)
        return current_pos  # Already finished
    
    def _check_capture(self, position):
        if position in self.start_positions:
            return None  # Starting positions are safe
        
        for player in range(4):
            if player != self.current_player:
                for piece in range(4):
                    if self.state[player][piece] == position:
                        # Capture occurred
                        self.state[player][piece] = -1  # Send piece back home
                        return (player, piece)
        
        return None  # No capture occurred
    
    def _get_obs(self):
        return {
            'board_state': self.state,
            'current_player': self.current_player,
            'last_roll': self.dice_roll
        } 
    
    def _get_info(self):
        return {i: {
            'player_pieces': self.state[i][:],
            'current_player': self.current_player,
            'last_roll': self.dice_roll
        } for i in range(4)}
    
    def render(self):
        print(f"Current state:")
        for i, player_pieces in enumerate(self.state):
            print(f"Player {i}: {player_pieces}")
        print(f"Current player: {self.current_player}")
        print(f"Last dice roll: {self.dice_roll}")

    def render_detailed(self):
        # Create an empty board
        board = [' ' for _ in range(52)]
        home_areas = [['H' for _ in range(4)] for _ in range(4)]
        home_stretches = [['.' for _ in range(6)] for _ in range(4)]

        # Fill the board with player pieces
        for player, pieces in enumerate(self.state):
            for i, pos in enumerate(pieces):
                if pos == -1:
                    home_areas[player][i] = str(i)
                elif 0 <= pos < 52:
                    board[pos] = f"{player}{i}"
                elif 52 <= pos < 58:
                    home_stretches[player][pos-52] = f"{player}{i}"

        # Create the board representation
        board_repr = [
            "    12 11 10        09 08 07    ",
            "    {} {} {}        {} {} {}    ",
            "13 {}           06 {}",
            "14 {}           05 {}",
            "15 {}           04 {}",
            "16 {}           03 {}",
            "17 {}           02 {}",
            "18 {}           01 {}",
            "    {} {} {}        {} {} {}    ",
            "    19 20 21        00 51 50    "
        ]

        # Fill in the board representation
        board_repr = [row.format(*[board[i] for i in range(52) if str(i) in row]) for row in board_repr]

        # Add home areas and home stretches
        home_area_repr = [
            "P0 Home: {}  P0 Stretch: {}",
            "P1 Home: {}  P1 Stretch: {}",
            "P2 Home: {}  P2 Stretch: {}",
            "P3 Home: {}  P3 Stretch: {}"
        ]

        home_area_repr = [home_area_repr[i].format(' '.join(home_areas[i]), ' '.join(home_stretches[i])) for i in range(4)]

        # Print the entire board
        print("\n".join(board_repr))
        print("\n".join(home_area_repr))
        print(f"Current player: {self.current_player}")
        print(f"Last dice roll: {self.dice_roll}")
# Register the environment
gym.register(
    id='MultiAgentLudo-v0',
    entry_point='MultiAgentLudoEnv:MultiAgentLudoEnv',
)


if __name__ == '__main__':
    env = gym.make('MultiAgentLudo-v0')
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()  # This would be your agent's choice
        obs, reward, done, truncated, info = env.step(action)
        env.render_detailed()
        if done:
            obs, info = env.reset()