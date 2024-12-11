from Ludo.envs import LudoEnv
from Ludo.utils.LudoPygameVisualizer import LudoPygameVisualizer
from time import sleep

RUN_VISUALIZER = True


class AdvancedLudoAgent:
    """
    An advanced Ludo agent that makes strategic decisions based on multiple factors
    including piece positions, opponent positions, and game state analysis.
    """

    def __init__(self):
        # Game constants
        self.OUT_OF_BOUNDS = -1
        self.START_SQUARE = 0
        self.FINAL_SQUARE = 58
        self.SAFE_POSITION = 52
        self.QUARTER_RUN = 13
        self.HOME_STRETCH = 50  # Position where pieces enter the final straight

        # Strategic weights for different aspects of the game
        self.WEIGHTS = {
            "get_out": 120,  # Weight for getting pieces into play
            "capture": 150,  # Weight for capturing opponent pieces
            "progress": 2,  # Weight for forward progress
            "finish": 250,  # Weight for reaching final square
            "safe_spot": 80,  # Weight for reaching safe spots
            "vulnerability": 70,  # Weight for avoiding vulnerable positions
            "blocking": 90,  # Weight for blocking opponents
            "home_stretch": 100,  # Weight for entering home stretch
            "grouping": 60,  # Weight for keeping pieces together
            "endgame": 180,  # Weight for endgame moves
        }

    def get_action(self, observation, action_space):
        """
        Determines the best action based on current game state.

        Args:
            observation: Dictionary containing board_state and last_roll
            action_space: The available actions

        Returns:
            int: Index of the piece to move
        """
        board_state = observation["observation"]["board_state"]
        dice_roll = observation["observation"]["last_roll"]
        current_player_pieces = board_state[0]
        opponent_pieces = board_state[1:]

        # Calculate scores for each possible move
        move_scores = []
        for piece_idx in range(len(current_player_pieces)):
            score = self._evaluate_move(
                piece_idx,
                current_player_pieces,
                opponent_pieces,
                board_state,
                dice_roll,
            )
            move_scores.append(score)

        # Return the index of the piece with highest score
        return max(range(len(move_scores)), key=lambda i: move_scores[i])

    def _evaluate_move(
        self, piece_idx, current_pieces, opponent_pieces, board_state, dice_roll
    ):
        """
        Evaluates the potential value of moving a specific piece.

        Args:
            piece_idx: Index of the piece to evaluate
            current_pieces: Array of current player's piece positions
            opponent_pieces: Array of opponent piece positions
            board_state: Complete board state
            dice_roll: Current dice roll value

        Returns:
            float: Score representing the move's value
        """
        current_pos = current_pieces[piece_idx]
        if current_pos == self.FINAL_SQUARE:
            return float("-inf")  # Don't move pieces that have finished

        new_pos = self._calculate_new_position(current_pos, dice_roll)
        score = 0

        # 1. Getting pieces into play
        if current_pos == self.OUT_OF_BOUNDS and dice_roll == 6:
            score += self.WEIGHTS["get_out"]
            # Additional bonus if other pieces are vulnerable
            if self._count_vulnerable_pieces(current_pieces) > 0:
                score += self.WEIGHTS["get_out"] * 0.5

        # 2. Evaluate forward progress
        if current_pos != self.OUT_OF_BOUNDS and new_pos <= self.FINAL_SQUARE:
            progress_score = new_pos * self.WEIGHTS["progress"]

            # Bonus for entering home stretch
            if current_pos < self.HOME_STRETCH and new_pos >= self.HOME_STRETCH:
                progress_score += self.WEIGHTS["home_stretch"]

            # Extra bonus for exact landing on FINAL_SQUARE
            if new_pos == self.FINAL_SQUARE:
                progress_score += self.WEIGHTS["finish"]

            score += progress_score

        # 3. Capture opportunities
        capture_value = self._evaluate_capture_opportunity(new_pos, opponent_pieces)
        score += capture_value * self.WEIGHTS["capture"]

        # 4. Safety evaluation
        if self._is_vulnerable(current_pos, opponent_pieces):
            score += self.WEIGHTS["vulnerability"]  # Encourage moving from danger

        # 5. Safe spot bonus
        if self._is_safe_spot(new_pos):
            score += self.WEIGHTS["safe_spot"]

        # 6. Strategic blocking
        blocking_value = self._evaluate_blocking(new_pos, opponent_pieces)
        score += blocking_value * self.WEIGHTS["blocking"]

        # 7. Piece grouping strategy
        grouping_value = self._evaluate_grouping(new_pos, current_pieces)
        score += grouping_value * self.WEIGHTS["grouping"]

        # 8. Endgame considerations
        if self._is_endgame(current_pieces):
            score += self._evaluate_endgame_move(new_pos) * self.WEIGHTS["endgame"]

        return score

    def _calculate_new_position(self, current_pos, dice_roll):
        """Calculates the new position after a move."""
        if current_pos == self.OUT_OF_BOUNDS:
            return self.START_SQUARE if dice_roll == 6 else self.OUT_OF_BOUNDS
        new_pos = current_pos + dice_roll
        return new_pos if new_pos <= self.FINAL_SQUARE else current_pos

    def _evaluate_capture_opportunity(self, position, opponent_pieces):
        """
        Evaluates the value of potential captures at the target position.
        Returns a value between 0 and 1.
        """
        if position == self.OUT_OF_BOUNDS or position == self.FINAL_SQUARE:
            return 0

        capture_value = 0
        for player_pieces in opponent_pieces:
            for piece_pos in player_pieces:
                if piece_pos == position:
                    capture_value += 1
                    # Extra value for capturing pieces closer to finish
                    capture_value += piece_pos / self.FINAL_SQUARE
        return min(capture_value, 1.0)

    def _is_vulnerable(self, position, opponent_pieces):
        """
        Checks if a piece is vulnerable to capture.
        Considers opponent pieces within striking distance.
        """
        if position == self.OUT_OF_BOUNDS or position == self.FINAL_SQUARE:
            return False

        for player_pieces in opponent_pieces:
            for piece_pos in player_pieces:
                if piece_pos != self.OUT_OF_BOUNDS and piece_pos != self.FINAL_SQUARE:
                    distance = position - piece_pos
                    if 0 < distance <= 6:  # Within dice roll range
                        return True
        return False

    def _is_safe_spot(self, position):
        """Identifies safe spots on the board."""
        return (
            position == self.START_SQUARE
            or position % self.QUARTER_RUN == 0
            or position >= self.SAFE_POSITION
        )

    def _evaluate_blocking(self, position, opponent_pieces):
        """
        Evaluates the strategic value of blocking opponent pieces.
        Returns a value between 0 and 1.
        """
        if position == self.OUT_OF_BOUNDS or position == self.FINAL_SQUARE:
            return 0

        blocking_value = 0
        for player_pieces in opponent_pieces:
            for piece_pos in player_pieces:
                if piece_pos != self.OUT_OF_BOUNDS and piece_pos < position:
                    distance = position - piece_pos
                    if distance < self.QUARTER_RUN:  # Close enough to be relevant
                        blocking_value += (
                            self.QUARTER_RUN - distance
                        ) / self.QUARTER_RUN
        return min(blocking_value, 1.0)

    def _evaluate_grouping(self, new_pos, current_pieces):
        """
        Evaluates the benefit of keeping pieces grouped together.
        Returns a value between 0 and 1.
        """
        if new_pos == self.OUT_OF_BOUNDS or new_pos == self.FINAL_SQUARE:
            return 0

        grouping_value = 0
        for pos in current_pieces:
            if pos != self.OUT_OF_BOUNDS and pos != self.FINAL_SQUARE:
                distance = abs(new_pos - pos)
                if 0 < distance <= 6:  # Within mutual support range
                    grouping_value += (6 - distance) / 6
        return min(grouping_value, 1.0)

    def _is_endgame(self, current_pieces):
        """
        Determines if the game is in endgame phase.
        """
        finished_pieces = sum(1 for pos in current_pieces if pos == self.FINAL_SQUARE)
        advanced_pieces = sum(1 for pos in current_pieces if pos >= self.HOME_STRETCH)
        return finished_pieces >= 2 or advanced_pieces >= 3

    def _evaluate_endgame_move(self, position):
        """
        Provides additional scoring for moves during endgame.
        Returns a value between 0 and 1.
        """
        if position >= self.HOME_STRETCH:
            return (position - self.HOME_STRETCH) / (
                self.FINAL_SQUARE - self.HOME_STRETCH
            )
        return 0

    def _count_vulnerable_pieces(self, current_pieces):
        """
        Counts how many pieces are in vulnerable positions.
        """
        return sum(
            1
            for pos in current_pieces
            if pos != self.OUT_OF_BOUNDS
            and pos != self.FINAL_SQUARE
            and not self._is_safe_spot(pos)
        )


# Usage in main game loop:
if __name__ == "__main__":
    env = LudoEnv()
    board = None
    if RUN_VISUALIZER:
        board = LudoPygameVisualizer()
    env.reset(seed=42)

    advanced_agent = AdvancedLudoAgent()

    for agent in env.agent_iter():
        if board is not None:
            board.update(env.board_state)
            sleep(0.1)
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} has reached the final square.")
            break

        # Use the advanced agent
        action = advanced_agent.get_action(observation, env.action_space(agent))
        env.step(int(action))
        env.render()
    env.close()