import pygame
import numpy as np

# Board layout constants
BOARD_SIZE = 15
MAIN_PATH_LENGTH = 52
TOTAL_POSITIONS = 58
HALF_PATH_LENGTH = 26
PLAYER_COUNT = 4
PIECES_PER_PLAYER = 4


class BoardDimensions:
    def __init__(self, size: int):
        self.total_size = size
        self.cell_size = size // BOARD_SIZE
        self.corner_offset = 2
        self.corner_size = 6
        self.home_size = 4
        self.piece_radius = self.cell_size // 3


class LudoPygameVisualizer:
    """Simple Pygame-based Ludo board visualizer."""

    PLAYER_COLORS = ["green", "red", "blue", "yellow"]
    COLORS = {
        "green": (76, 175, 80),
        "red": (244, 67, 54),
        "blue": (33, 150, 243),
        "yellow": (255, 235, 59),
        "white": (255, 255, 255),
        "black": (33, 33, 33),
        "grid": (200, 200, 200),
    }

    def __init__(self, size: int = 600):
        pygame.init()
        self.dimensions = BoardDimensions(size)
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Ludo Game")

        # Step pattern represents distance from center for each position
        self.steps = [1] * 6 + [2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 2] + [1] * 6 + [0]

        # Precompute positions
        self.home_positions = self._init_home_positions()
        self.path_positions = self._init_path_positions()
        self.starting_positions = self._init_starting_positions()
        self.final_positions = self._init_final_positions()

    def _init_home_positions(self) -> list[list[tuple[int, int]]]:
        """Initialize starting positions for each player's pieces."""
        corners = [(2, 2), (11, 2), (11, 11), (2, 11)]
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        return [
            [(cor_x + off_x, cor_y + off_y) for off_x, off_y in offsets]
            for cor_x, cor_y in corners
        ]

    def _init_starting_positions(self) -> list[tuple[int, int]]:
        """Initialize starting positions for each player."""
        return [(1, 6), (8, 1), (13, 8), (6, 13)]

    def _init_path_positions(self) -> list[tuple[int, int]]:
        """Initialize the main path coordinates using the same logic as SVG version."""
        positions = []
        for i in range(MAIN_PATH_LENGTH + 1):
            point = self._index_to_grid(i)
            x = int(point.real) + 7
            y = int(point.imag) + 7
            positions.append((x, y))
        return positions

    def _calculate_coordinate_signs(
        self, x_index: int, y_index: int
    ) -> tuple[int, int]:
        y_sign = 1 if y_index >= HALF_PATH_LENGTH else -1
        x_sign = 1 if x_index // HALF_PATH_LENGTH % 2 == 0 else -1
        return x_sign, y_sign

    def _index_to_grid(self, index: int) -> complex:
        index = (index + 1) % MAIN_PATH_LENGTH
        x_index, y_index = index - 13, index
        x_sign, y_sign = self._calculate_coordinate_signs(x_index, y_index)
        return complex(
            x_sign * self.steps[x_index % HALF_PATH_LENGTH],
            y_sign * self.steps[y_index % HALF_PATH_LENGTH],
        )

    def _init_final_positions(self) -> list[list[tuple[int, int]]]:
        """Initialize home stretch positions for each player."""
        return [
            [(i, 7) for i in range(1, 7)],  # Green
            [(7, i) for i in range(1, 7)],  # Red
            [(13 - i, 7) for i in range(6)],  # Blue
            [(7, 13 - i) for i in range(6)],  # Yellow
        ]

    def _draw_corner_square(self, pos: tuple[int, int], color: str) -> None:
        """Draw a single corner square with nested rectangles."""
        x, y = pos
        d = self.dimensions

        # Outer black rectangle
        pygame.draw.rect(
            self.screen,
            self.COLORS["black"],
            (
                (x - d.corner_offset) * d.cell_size,
                (y - d.corner_offset) * d.cell_size,
                d.corner_size * d.cell_size,
                d.corner_size * d.cell_size,
            ),
        )

        # Colored rectangle
        pygame.draw.rect(
            self.screen,
            self.COLORS[color],
            (
                (x - 1) * d.cell_size,
                (y - 1) * d.cell_size,
                d.home_size * d.cell_size,
                d.home_size * d.cell_size,
            ),
        )

        # Inner white rectangle
        pygame.draw.rect(
            self.screen,
            self.COLORS["white"],
            (
                x * d.cell_size,
                y * d.cell_size,
                2 * d.cell_size,
                2 * d.cell_size,
            ),
        )

    def draw_board(self) -> None:
        """Draw the basic board layout."""
        self.screen.fill(self.COLORS["white"])

        # Draw corner squares
        for i, color in enumerate(self.PLAYER_COLORS):
            self._draw_corner_square(self.home_positions[i][0], color)

        self._draw_grid()
        self._draw_center_arrows()
        self._draw_starting_positions()
        self._draw_final_positions()

    def _draw_grid(self) -> None:
        """Draw the grid lines."""
        for i in range(BOARD_SIZE):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (0, i * self.dimensions.cell_size),
                (self.dimensions.total_size, i * self.dimensions.cell_size),
            )
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (i * self.dimensions.cell_size, 0),
                (i * self.dimensions.cell_size, self.dimensions.total_size),
            )

    def _draw_center_arrows(self) -> None:
        """Draw the center arrow design."""
        center = (7.5, 7.5)
        triangles = [
            [(6, 6), (6, 9), center],
            [(6, 6), (9, 6), center],
            [(9, 6), (9, 9), center],
            [(9, 9), (6, 9), center],
        ]

        for color, triangle in zip(self.PLAYER_COLORS, triangles):
            scaled_points = [
                (x * self.dimensions.cell_size, y * self.dimensions.cell_size)
                for x, y in triangle
            ]
            pygame.draw.polygon(self.screen, self.COLORS[color], scaled_points)

    def _draw_starting_positions(self) -> None:
        """Draw the starting position squares."""
        for color, (cell_x, cell_y) in zip(self.PLAYER_COLORS, self.starting_positions):
            cell_points = self._get_cell_points(cell_x, cell_y)
            pygame.draw.polygon(self.screen, self.COLORS[color], cell_points)

    def _get_cell_points(self, cell_x: int, cell_y: int) -> list[tuple[int, int]]:
        """Helper method to calculate cell corner points."""
        return [
            (cell_x * self.dimensions.cell_size, cell_y * self.dimensions.cell_size),
            (
                (cell_x + 1) * self.dimensions.cell_size,
                cell_y * self.dimensions.cell_size,
            ),
            (
                (cell_x + 1) * self.dimensions.cell_size,
                (cell_y + 1) * self.dimensions.cell_size,
            ),
            (
                cell_x * self.dimensions.cell_size,
                (cell_y + 1) * self.dimensions.cell_size,
            ),
        ]

    def _draw_final_positions(self) -> None:
        """Draw the final position squares."""
        for color, cells in zip(self.PLAYER_COLORS, self.final_positions):
            for cell in cells:
                cell_points = self._get_cell_points(cell[0], cell[1])
                pygame.draw.polygon(self.screen, self.COLORS[color], cell_points)

    def update(self, board_state: np.ndarray) -> None:
        """Update the display with current game state."""
        assert board_state.shape == (PLAYER_COUNT, PIECES_PER_PLAYER)
        self.draw_board()
        self.draw_pieces(board_state)
        pygame.display.flip()

    def close(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()

    def _calculate_piece_position(
        self, player: int, piece: int, pos: int
    ) -> tuple[int, int]:
        """Calculate the grid position for a piece based on its state."""
        if pos not in range(-1, 59):
            raise ValueError(f"Invalid position: {pos}")
        if pos == -1:  # Home position
            return self.home_positions[player][piece]
        elif pos > MAIN_PATH_LENGTH:  # Final stretch
            final_index = pos - MAIN_PATH_LENGTH - 1
            return self.final_positions[player][final_index]

        else:  # Main path
            adjusted_pos = (pos + 13 * player) % MAIN_PATH_LENGTH
            return self.path_positions[adjusted_pos]

    def _draw_piece(self, x: int, y: int, color: str) -> None:
        """Draw a single game piece at the specified position."""
        center_x = x * self.dimensions.cell_size + self.dimensions.cell_size // 2
        center_y = y * self.dimensions.cell_size + self.dimensions.cell_size // 2

        pygame.draw.circle(
            self.screen,
            self.COLORS[color],
            (center_x, center_y),
            self.dimensions.piece_radius,
        )
        pygame.draw.circle(
            self.screen,
            self.COLORS["black"],
            (center_x, center_y),
            self.dimensions.piece_radius,
            width=1,
        )

    def draw_pieces(self, board_state: np.ndarray) -> None:
        """Draw all pieces based on their current positions."""
        # Position types: -1=home, 0-52=main path, 53+=final stretch
        for player in range(PLAYER_COUNT):
            for piece in range(PIECES_PER_PLAYER):
                pos = board_state[player][piece]
                piece_pos = self._calculate_piece_position(player, piece, pos)

                if piece_pos:
                    self._draw_piece(*piece_pos, self.PLAYER_COLORS[player])


if __name__ == "__main__":
    # Create visualizer
    viz = LudoPygameVisualizer()

    # Example board state - all pieces in starting positions
    board = np.full((PLAYER_COUNT, PIECES_PER_PLAYER), -1, dtype=np.int8)

    # Main game loop
    i = 0
    while True:
        i += 1
        player_and_piece = i % (PLAYER_COUNT * PIECES_PER_PLAYER)
        piece_index, player_index = divmod(player_and_piece, PIECES_PER_PLAYER)
        viz.update(board)
        dice_roll = 1
        board[player_index, piece_index] += dice_roll

        if board[player_index, piece_index] > TOTAL_POSITIONS:
            break
        pygame.time.wait(100)  # Cap at 10 FPS

    viz.close()
