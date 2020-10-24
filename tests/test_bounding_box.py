from pathlib import Path

import numpy as np
from PIL import Image

from src.bounding_box import get_bounding_box, non_max_suppression


def test_non_max_suppression_empty_input():
    assert [] == non_max_suppression([], 0.5), "Wrong return value"


def test_get_bounding_box():
    image = Image.open(str(Path.cwd() / "data" / "equation_example.jpg"))

    bounding_boxes = get_bounding_box(np.array(image))

    expected_boxes = [
        np.array([76, 94, 10, 49]),
        np.array([113, 94, 45, 49]),
        np.array([163, 94, 49, 49]),
        np.array([216, 101, 49, 35]),
        np.array([280, 94, 28, 49]),
        np.array([322, 112, 49, 11]),
        np.array([377, 94, 44, 49]),
        np.array([428, 96, 49, 45]),
        np.array([479, 102, 51, 33]),
        np.array([545, 94, 27, 49]),
    ]
    assert all(
        np.all(expected == actual)
        for expected, actual in zip(expected_boxes, bounding_boxes)
    ), bounding_boxes
