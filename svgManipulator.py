import xml.etree.ElementTree as ET


class LudoVisualizer:
    def __init__(self, svg_file):
        with open(svg_file, "r") as file:
            self.svg_string = file.read()
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self.root = ET.fromstring(self.svg_string)
        self.players = list(self.root[12][4:])
        self.offset = 7 + 7j
        self.steps = [1] * 6 + [2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3, 2] + [1] * 6 + [0]
        offsets = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
        self.outofbounds = [
            [2 + 2j + offset + 9 * player_offset for offset in offsets]
            for player_offset in offsets  # generates coor of 4 pieces in 4 corners
        ]
        self.end = [
            [(self.index_to_grid(0) + 1j + i) * 1j**p + self.offset for i in range(6)]
            for p in range(4)  # p is player index
        ]

    def move_piece(self, player_index: int, piece_num: int, piece_index: int):
        if player_index not in range(4):
            raise ValueError(f"Invalid player index. Use 0-3.{player_index}")
        if piece_num not in range(4):
            raise ValueError(f"Invalid piece number. Use 0-3.{piece_num}")
        if piece_index not in range(-1, 59):
            raise ValueError(f"Invalid piece index. Use [-1,58].{piece_index}")
        
        coordinate = self.outofbounds[player_index][piece_num]
        if piece_index in range(53):
            coordinate = self.calc_position(player_index, piece_index)
        elif piece_index in range(53, 59):
            coordinate = self.end[player_index][piece_index - 53]


        i, j = complex(coordinate).real, complex(coordinate).imag

        circle = self.players[player_index][piece_num]

        new_x = i * 10 + 5
        new_y = j * 10 + 5

        circle.set("cx", str(new_x))
        circle.set("cy", str(new_y))

    def index_to_grid(self, index: int) -> complex:
        index = (index + 1) % 52
        real_index, imag_index = index - 13, index
        img_sign = 1 if imag_index >= 26 else -1
        real_sign = 1 if real_index // 26 % 2 == 0 else -1

        return complex(
            real_sign * self.steps[real_index % 26],
            img_sign * self.steps[imag_index % 26],
        )

    def coordinate_on_board(self, index: int) -> complex:
        return self.index_to_grid(index) + self.offset

    def calc_position(self, player_index: int, piece_index: int) -> complex:
        return self.coordinate_on_board(piece_index + 13 * player_index)

    def get_svg_string(self):
        return ET.tostring(self.root, encoding="unicode")

    def save_svg(self, filename):
        with open(filename, "w") as file:
            file.write(self.get_svg_string())


if __name__ == "__main__":
    # Usage
    game = LudoVisualizer("ludo.svg")
    # game.move_piece(0, 0, 52)
    # game.save_svg("ludo_new.svg")
    # game.display()
