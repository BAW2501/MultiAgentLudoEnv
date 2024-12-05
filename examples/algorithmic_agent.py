from Ludo.envs import LudoEnv
from Ludo.utils.LudoPygameVisualizer import LudoPygameVisualizer
from time import sleep

RUN_VISUALIZER = True
class SmartLudoAgent:
    def __init__(self):
        self.OUT_OF_BOUNDS = -1
        self.START_SQUARE = 0
        self.FINAL_SQUARE = 58
        self.SAFE_POSITION = 52
        self.QUARTER_RUN = 13

    def get_action(self, observation, action_space):
        board_state = observation["observation"]["board_state"]
        dice_roll = observation["observation"]["last_roll"]
        current_player_pieces = board_state[0]  # Current player's pieces
        
        # Calculate scores for each possible move
        move_scores = []
        for piece_idx in range(len(current_player_pieces)):
            score = self._evaluate_move(piece_idx, current_player_pieces, board_state, dice_roll)
            move_scores.append(score)
        
        # Choose the piece with the highest score
        best_piece = max(range(len(move_scores)), key=lambda i: move_scores[i])
        return best_piece

    def _evaluate_move(self, piece_idx, current_pieces, board_state, dice_roll):
        current_pos = current_pieces[piece_idx]
        if current_pos == self.FINAL_SQUARE:
            return -1000  # Don't move pieces that have reached the end
        
        new_pos = self._calculate_new_position(current_pos, dice_roll)
        score = 0
        
        # Priority 1: Getting pieces out if possible
        if current_pos == self.OUT_OF_BOUNDS and dice_roll == 6:
            score += 100
            
        # Priority 2: Moving pieces close to finishing
        if current_pos != self.OUT_OF_BOUNDS and new_pos <= self.FINAL_SQUARE:
            score += new_pos * 2
            
            # Extra points for exact landing on FINAL_SQUARE
            if new_pos == self.FINAL_SQUARE:
                score += 200
                
        # Priority 3: Capture opportunities
        if self._can_capture_opponent(new_pos, board_state):
            score += 150
            
        # Priority 4: Avoid being captured
        if self._is_vulnerable(current_pos, board_state):
            score += 50  # Encourage moving from vulnerable positions
            
        # Priority 5: Prefer safe spots
        if self._is_safe_spot(new_pos):
            score += 75
            
        return score

    def _calculate_new_position(self, current_pos, dice_roll):
        if current_pos == self.OUT_OF_BOUNDS:
            return self.START_SQUARE if dice_roll == 6 else self.OUT_OF_BOUNDS
        new_pos = current_pos + dice_roll
        return new_pos if new_pos <= self.FINAL_SQUARE else current_pos

    def _can_capture_opponent(self, position, board_state):
        if position == self.OUT_OF_BOUNDS or position == self.FINAL_SQUARE:
            return False
        
        # Check if any opponent pieces are at this position
        for player_idx in range(1, len(board_state)):
            for piece_pos in board_state[player_idx]:
                if piece_pos == position:
                    return True
        return False

    def _is_vulnerable(self, position, board_state):
        if position == self.OUT_OF_BOUNDS or position == self.FINAL_SQUARE:
            return False
        
        # Check if any opponent pieces are within 6 spaces
        for player_idx in range(1, len(board_state)):
            for piece_pos in board_state[player_idx]:
                if piece_pos != self.OUT_OF_BOUNDS and piece_pos != self.FINAL_SQUARE:
                    if 0 < (position - piece_pos) <= 6:
                        return True
        return False

    def _is_safe_spot(self, position):
        # Define safe spots (starting positions and every 13th square)
        return position == self.START_SQUARE or position % self.QUARTER_RUN == 0

# Usage in main game loop:
if __name__ == "__main__":
    env = LudoEnv()
    board = None
    if RUN_VISUALIZER:
        board = LudoPygameVisualizer()
    env.reset(seed=42)
    
    smart_agent = SmartLudoAgent()

    for agent in env.agent_iter():
        if board is not None:
            board.update(env.board_state)
            sleep(0.5)
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} has reached the final square.")
            break
            
        # Use the smart agent instead of random action
        action = smart_agent.get_action(observation, env.action_space(agent))
        env.step(int(action))
        env.render()
    env.close()