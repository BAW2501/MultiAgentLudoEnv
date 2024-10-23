import pygame
import numpy as np


class LudoPygameVisualizer:
    """Simple Pygame-based Ludo board visualizer."""

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
        self.size = size
        self.cell_size = size // 15
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Ludo Game")

        # Define step pattern similar to SVG version
        self.steps = [1] * 6 + [2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 2] + [1] * 6 + [0]

        # Precompute positions
        self.home_positions = self._init_home_positions()
        self.path_positions = self._init_path_positions()
        self.starting_positions = self._init_starting_positions()
        self.final_positions = self._init_final_positions()

    def _init_home_positions(self) -> list[list[tuple[int, int]]]:
        """Initialize starting positions for each player's pieces."""
        corners = [(2, 2), (2, 11), (11, 2), (11, 11)]
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        return [
            [(corner[0] + offset[0], corner[1] + offset[1]) for offset in offsets]
            for corner in corners
        ]

    def _init_starting_positions(self) -> list[tuple[int, int]]:
        """Initialize starting positions for each player."""
        return [(1, 6), (8, 1), (13, 8), (6, 13)]

    def _init_path_positions(self) -> list[tuple[int, int]]:
        """Initialize the main path coordinates using the same logic as SVG version."""
        positions = []
        for i in range(52):
            point = self._index_to_grid(i)
            # Adjust coordinates to match Pygame grid
            x = int(point.real) + 7
            y = int(point.imag) + 7
            positions.append((x, y))
        return positions
    def _calculate_coordinate_signs(
        self, real_index: int, imag_index: int
    ) -> tuple[int, int]:
        imag_sign = 1 if imag_index >= 26 else -1
        real_sign = 1 if real_index // 26 % 2 == 0 else -1
        return real_sign, imag_sign
    def _index_to_grid(self, index: int) -> complex:
        # Convert linear index to 2D grid coordinate
        index = (index + 1) % 52
        real_index, imag_index = index - 13, index
        real_sign, imag_sign = self._calculate_coordinate_signs(real_index, imag_index)
        return complex(
            real_sign * self.steps[real_index % 26],
            imag_sign * self.steps[imag_index % 26],
        )
        
    def _init_final_positions(self) -> list[list[tuple[int, int]]]:
        """Initialize home stretch positions for each player."""
        return [
            [(i, 7) for i in range(1, 7)],  # Green
            [(7, i) for i in range(1, 7)],  # Red
            [(13 - i, 7) for i in range(6)],  # Blue
            [(7, 13 - i) for i in range(6)],  # Yellow
        ]

    def draw_board(self) -> None:
        """Draw the basic board layout."""
        self.screen.fill(self.COLORS["white"])

        # Draw corner squares
        for i, color in enumerate(["green", "red", "yellow", "blue"]):
            x, y = self.home_positions[i][0]
            pygame.draw.rect(
                self.screen,
                self.COLORS["black"],
                (
                    (x - 2) * self.cell_size,
                    (y - 2) * self.cell_size,
                    6 * self.cell_size,
                    6 * self.cell_size,
                ),
            )

            pygame.draw.rect(
                self.screen,
                self.COLORS[color],
                (
                    (x - 1) * self.cell_size,
                    (y - 1) * self.cell_size,
                    4 * self.cell_size,
                    4 * self.cell_size,
                ),
            )

            pygame.draw.rect(
                self.screen,
                self.COLORS["white"],
                (
                    x * self.cell_size,
                    y * self.cell_size,
                    2 * self.cell_size,
                    2 * self.cell_size,
                ),
            )

        # Draw grid and other board elements
        self._draw_grid()
        self._draw_center_arrows()
        self._draw_starting_positions()
        self._draw_final_positions()

    def _draw_grid(self) -> None:
        """Draw the grid lines."""
        for i in range(15):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (0, i * self.cell_size),
                (self.size, i * self.cell_size),
            )
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.size),
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

        for color, triangle in zip(["green", "red", "blue", "yellow"], triangles):
            scaled_points = [
                (x * self.cell_size, y * self.cell_size) for x, y in triangle
            ]
            pygame.draw.polygon(self.screen, self.COLORS[color], scaled_points)

    def _draw_starting_positions(self) -> None:
        """Draw the starting position squares."""
        for color, (cell_x, cell_y) in zip(
            ["green", "red", "blue", "yellow"], self.starting_positions
        ):
            cell_points = [
                (cell_x * self.cell_size, cell_y * self.cell_size),
                ((cell_x + 1) * self.cell_size, cell_y * self.cell_size),
                ((cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size),
                (cell_x * self.cell_size, (cell_y + 1) * self.cell_size),
            ]
            pygame.draw.polygon(self.screen, self.COLORS[color], cell_points)

    def _draw_final_positions(self) -> None:
        """Draw the final position squares."""
        for color, cells in zip(
            ["green", "red", "blue", "yellow"], self.final_positions
        ):
            for cell in cells:
                cell_points = [
                    (cell[0] * self.cell_size, cell[1] * self.cell_size),
                    ((cell[0] + 1) * self.cell_size, cell[1] * self.cell_size),
                    ((cell[0] + 1) * self.cell_size, (cell[1] + 1) * self.cell_size),
                    (cell[0] * self.cell_size, (cell[1] + 1) * self.cell_size),
                ]
                pygame.draw.polygon(self.screen, self.COLORS[color], cell_points)

    def draw_pieces(self, board_state: np.ndarray) -> None:
        """Draw all pieces based on their current positions."""
        colors = ["green", "red", "blue", "yellow"]
    
        for player in range(4):
            for piece in range(4):
                pos = board_state[player][piece]
                
                if pos == -1:  # Out of bounds
                    x, y = self.home_positions[player][piece]
                elif pos >= 52:  # In final stretch
                    final_index = pos - 52
                    if final_index < len(self.final_positions[player]):
                        x, y = self.final_positions[player][final_index]
                    else:
                        continue  # Skip if position is invalid
                else:  # On main path
                    adjusted_pos = (pos + 13 * player) % 52
                    x, y = self.path_positions[adjusted_pos]

                pygame.draw.circle(
                    self.screen,
                    self.COLORS[colors[player]],
                    (
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2,
                    ),
                    self.cell_size // 3,
                )
                pygame.draw.circle(
                    self.screen,
                    self.COLORS['black'],
                    (
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2,
                    ),
                    self.cell_size // 3,
                    width=1
                )

    def update(self, board_state: np.ndarray) -> None:
        """Update the display with current game state."""
        self.draw_board()
        self.draw_pieces(board_state)
        pygame.display.flip()

    def close(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()


if __name__ == "__main__":
    # Create visualizer
    viz = LudoPygameVisualizer()

    # Example board state
    board = np.full((4, 4), -1, dtype=np.int8)  # All pieces in starting positions

    # Main game loop
    i = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        viz.update(board)
        dice_roll = 1 #np.random.randint(1, 6)
        current_pos = board[0, 0]
        next_pos = (current_pos + dice_roll) % 58
        board[0, 0] = next_pos
        
        i += 1
        pygame.time.wait(100)  # Cap at 10 FPS

    #viz.close()