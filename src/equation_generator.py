"""This module contains an EquationGenerator class to generate an
equation by randomizing the image used for each character from the
data folder
"""
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
    """This class loads the file paths for each of the ALLOWED_CHARS
    and generates an equation with each of the character images chosen
    randomly. This class assumes the following data directory
    structure:

        self.data_dir
        +---"+"
        |   +---<"+" character image 1>.jpg
        |   +---<"+" character image 2>.jpg
        |   +---...
        +---"-"
        |   +---<"-" character image 1>.jpg
        |   +---<"-" character image 2>.jpg
        |   +---...
        +---<Other characters in ALLOWED_CHARS>

    Attributes:
        char_path_map (dict): Map characters to their corresponding
            image paths
        data_dir (pathlib.Path): Path to the data directory containing
            the character images
        output_size (tuple of int): The minimum output size of the
            equation image
        char_spacing (int): Spacing between each character image in the
            equation image
    """

    def __init__(self, data_dir, output_size, char_spacing=0):
        self.char_path_map = dict()
        self.data_dir = data_dir
        self.output_size = output_size
        self.char_spacing = char_spacing
        self.load_filenames()

    def load_filenames(self):
        """Load all character images with extensions from
        ACCEPTED_FORMATS from each of the character sub-directories in
        the data directory
        """
        for char in ALLOWED_CHARS.values():
            subdir = self.data_dir / char
            image_paths = []
            for ext in ACCEPTED_FORMATS:
                image_paths.extend(glob.glob(str(subdir / f"*.{ext}")))
            self.char_path_map[char] = image_paths

    def generate_character(self, char):
        """Generate an image of the character by randomly choosing it
        from corresponding data sub-directory

        Args:
            char (str): The character to generate

        Returns:
            PIL.Image: The character image
        """
        image_path = random.choice(self.char_path_map[ALLOWED_CHARS[char]])
        return Image.open(image_path)

    def generate_equation(self, equation):
        """Generate an equation image by randomly picking images for
        each character from the corresponding data sub-directory.

        Args:
            equation (str): The equation to generate

        Returns:
            PIL.Image: The equation image. The minimum image size is
                equation to self.output_size
            np.ndarray of int: The coordinates of each character in the
                equation. The coordinate format is
                [left, right, top, bottom]
            str: The original equation string
        """
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
