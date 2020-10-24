from pathlib import Path

import numpy as np
from PIL import Image

from src.equation_generator import EquationGenerator

ALLOWED_CHARS = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]
ALLOWED_CHARS = {char: char for char in ALLOWED_CHARS}
ALLOWED_CHARS["/"] = "div"
ALLOWED_CHARS["*"] = "times"
CHAR_SPACING = 8
DATA_DIR = Path.cwd() / "data" / "example"
OUTPUT_SIZE = (640, 236)


def test_equation_generator_constructor():
    generator = EquationGenerator(DATA_DIR, OUTPUT_SIZE, CHAR_SPACING)

    assert generator.data_dir == DATA_DIR, "Wrong data_dir"
    assert generator.output_size == OUTPUT_SIZE, "Wrong output_size"
    assert generator.char_spacing == CHAR_SPACING, "Wrong char_spacing"


def test_equation_generator_load_filenames():
    generator = EquationGenerator(DATA_DIR, OUTPUT_SIZE, CHAR_SPACING)

    for char in ALLOWED_CHARS.values():
        assert char in generator.char_path_map, f"{char} not found"
        assert generator.char_path_map[char] == [
            str(DATA_DIR / char / f"{char}_example.jpg")
        ], "Character image not found"


def test_equation_generator_generate_character():
    generator = EquationGenerator(DATA_DIR, OUTPUT_SIZE, CHAR_SPACING)

    for char in ALLOWED_CHARS:
        assert np.all(
            Image.open(
                DATA_DIR / ALLOWED_CHARS[char] / f"{ALLOWED_CHARS[char]}_example.jpg"
            )
            == generator.generate_character(char)
        ), "Wrong image opened"


def test_equation_generator_generate_equation():
    generator = EquationGenerator(DATA_DIR, OUTPUT_SIZE, CHAR_SPACING)

    expected_equation = ALLOWED_CHARS.keys()
    char_images = [generator.generate_character(char) for char in expected_equation]
    max_height = max(image.height for image in char_images)
    total_width = (
        sum(image.width for image in char_images)
        + (len(expected_equation) - 1) * CHAR_SPACING
    )
    expected_image = Image.new(
        "L",
        (max(total_width, OUTPUT_SIZE[0]), max(max_height, OUTPUT_SIZE[1])),
        255,
    )
    left = (expected_image.width - total_width) // 2
    expected_coords = []
    for image in char_images:
        top = expected_image.height // 2 - image.height // 2
        expected_image.paste(image, box=(left, top))
        left += image.width + CHAR_SPACING
        expected_coords.append([left, left + image.width, top, top + image.height])

    image, coords, equation = generator.generate_equation(expected_equation)

    assert expected_image == image, "Wrong image"
    assert np.all(np.array(expected_coords) == coords), "Wrong coords"
    assert expected_equation == equation, "Wrong equation"
