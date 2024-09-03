import xml.etree.ElementTree as ET

class LudoVisualizer:
    def __init__(self, svg_file):
        with open(svg_file, "r") as file:
            self.svg_string = file.read()
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self.root = ET.fromstring(self.svg_string)
        self.players = list(self.root[12][4:])

    def move_piece(self, player_index, piece_num, x, y):
        if piece_num not in range(4):
            raise ValueError("Invalid piece number. Use 0-3.")

        if x not in range(15) or y not in range(15):
            raise ValueError("Invalid grid coordinates. Use 0-14 for both x and y.")

        circle = self.players[player_index][piece_num]

        new_x = x * 10 + 5
        new_y = y * 10 + 5

        circle.set("cx", str(new_x))
        circle.set("cy", str(new_y))

    def get_svg_string(self):
        return ET.tostring(self.root, encoding="unicode")

    def save_svg(self, filename):
        with open(filename, "w") as file:
            file.write(self.get_svg_string())

if __name__ == "__main__":
    # Usage
    game = LudoVisualizer("ludo.svg")
    game.move_piece(0, 0, 4, 6)
    # game.save_svg("ludo_new.svg")
    # game.display()

