import glob
import random

import numpy as np
from PIL import Image


ALLOWED_CHARS = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]
ALLOWED_CHARS = {char: char for char in ALLOWED_CHARS}
ALLOWED_CHARS["/"] = "div"
ALLOWED_CHARS["*"] = "times"
ACCEPTED_FORMATS = set(["jpg"])


class EquationGenerator:
    def __init__(self, data_dir, output_size, char_spacing=0):
        self.char_path_map = dict()
        self.data_dir = data_dir
        self.output_size = output_size
        self.char_spacing = char_spacing
        self.load_filenames()

    def load_filenames(self):
        for char in ALLOWED_CHARS.values():
            subdir = self.data_dir / char
            image_paths = []
            for ext in ACCEPTED_FORMATS:
                image_paths.extend(glob.glob(str(subdir / f"*.{ext}")))
            self.char_path_map[char] = image_paths

    def generate_character(self, char):
        image_path = random.choice(self.char_path_map[ALLOWED_CHARS[char]])
        return Image.open(image_path)

    def generate_equation(self, equation):
        char_images = [self.generate_character(char) for char in equation]
        max_height = max(image.height for image in char_images)
        total_width = (
            sum(image.width for image in char_images)
            + (len(equation) - 1) * self.char_spacing
        )
        background = Image.new(
            "L",
            (
                max(total_width, self.output_size[0]),
                max(max_height, self.output_size[1]),
            ),
            255,
        )
        left = (background.width - total_width) // 2
        coords = []
        for image in char_images:
            top = background.height // 2 - image.height // 2
            background.paste(image, box=(left, top))
            left += image.width + self.char_spacing
            coords.append([left, left + image.width, top, top + image.height])
        return background, np.array(coords), equation
