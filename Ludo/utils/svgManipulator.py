import xml.etree.ElementTree as ET
from pathlib import Path


class LudoVisualizer:
    # Constants for board dimensions and player counts
    NUM_PLAYERS = 4
    NUM_END_POSITIONS = 6
    CELL_SIZE = 10
    CELL_CENTER_OFFSET = 5
    OFFSET = 7 + 7j  # svg is not centered on 0,0

    def __init__(self, filename: str = "ludo.svg") -> None:
        file_root = Path(__file__).parent / filename
        print(f"Loading SVG from {file_root}")
        self.steps = [1] * 6 + [2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 2] + [1] * 6 + [0]
        self._initialize_svg_root(file_root)  # load the svg to manipulate
        self._initialize_out_of_bounds()  # where to place pieces when out of the game
        self._initialize_end_positions()  # colored squares except starting one
        self.players = list(self.root[12][4:])  # Initialize player positions from SVG

    def _initialize_svg_root(self, svg_filename: str) -> None:
        with open(svg_filename, "r") as svg_file:
            self.svg_string = svg_file.read()
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self.root = ET.fromstring(self.svg_string)

    def _initialize_out_of_bounds(self) -> None:
        offsets = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]  # square, Right, Bottom, R & B
        self.outofbounds = [
            [2 + 2j + offset + 9 * player_offset for offset in offsets]
            for player_offset in offsets
        ]

    def _initialize_end_positions(self) -> None:
        self.end = [
            [
                (self.index_to_grid(0) + 1j + i) * 1j**p + self.OFFSET
                for i in range(self.NUM_END_POSITIONS)
            ]
            for p in range(self.NUM_PLAYERS)
        ]

    def _validate_move_parameters(
        self, player_index: int, piece_num: int, piece_index: int
    ) -> None:
        if player_index not in range(self.NUM_PLAYERS):
            raise ValueError(f"Invalid player index. Use 0-3. {player_index}")
        if piece_num not in range(self.NUM_PLAYERS):
            raise ValueError(f"Invalid piece number. Use 0-3. {piece_num}")
        if piece_index not in range(-1, 59):
            raise ValueError(f"Invalid piece index. Use [-1,58]. {piece_index}")

    def move_piece(self, player_index: int, piece_num: int, piece_index: int) -> None:
        self._validate_move_parameters(player_index, piece_num, piece_index)
        coordinate = self._get_piece_coordinate(player_index, piece_num, piece_index)
        x, y = coordinate.real, coordinate.imag

        circle = self.players[player_index][piece_num]
        new_x = x * self.CELL_SIZE + self.CELL_CENTER_OFFSET
        new_y = y * self.CELL_SIZE + self.CELL_CENTER_OFFSET

        circle.set("cx", str(new_x))
        circle.set("cy", str(new_y))

    def _get_piece_coordinate(
        self, player_index: int, piece_num: int, piece_index: int
    ) -> complex:
        if piece_index == -1:
            return self.outofbounds[player_index][piece_num]
        elif piece_index < 53:
            return self.calculate_position(player_index, piece_index)
        else:
            return self.end[player_index][piece_index - 53]

    def index_to_grid(self, index: int) -> complex:
        # Convert linear index to 2D grid coordinate
        index = (index + 1) % 52
        real_index, imag_index = index - 13, index
        real_sign, imag_sign = self._calculate_coordinate_signs(real_index, imag_index)
        return complex(
            real_sign * self.steps[real_index % 26],
            imag_sign * self.steps[imag_index % 26],
        )

    def _calculate_coordinate_signs(
        self, real_index: int, imag_index: int
    ) -> tuple[int, int]:
        imag_sign = 1 if imag_index >= 26 else -1
        real_sign = 1 if real_index // 26 % 2 == 0 else -1
        return real_sign, imag_sign

    def coordinate_on_board(self, index: int) -> complex:
        return self.index_to_grid(index) + self.OFFSET

    def calculate_position(self, player_index: int, piece_index: int) -> complex:
        return self.coordinate_on_board(piece_index + 13 * player_index)

    def set_board_from_array(self, board: list[list[int]]) -> None:
        if len(board) != self.NUM_PLAYERS or any(
            len(player) != self.NUM_PLAYERS for player in board
        ):
            raise ValueError("Input must be a 4x4 array of arrays")

        for player_index, player_pieces in enumerate(board):
            for piece_num, piece_index in enumerate(player_pieces):
                self.move_piece(player_index, piece_num, piece_index)

    def get_svg_string(self) -> str:
        # Convert SVG tree to string representation
        return ET.tostring(self.root, encoding="unicode")

    def save_svg(self, filename: str) -> None:
        # Save current SVG state to file
        with open(filename, "w") as file:
            file.write(self.get_svg_string())
