from pathlib import Path

import cv2
import numpy as np

from src.utils.structural import (
    approximate_contour,
    get_edged_image,
    get_sorted_contours,
)

IMAGE = cv2.imread(str(Path.cwd() / "data" / "example_filled.jpg"))


def test_approximate_contour():
    expected = np.array([[[3, 2]], [[11, 4]], [[4, 21]], [[10, 15]]])
    contour = np.copy(expected)

    num_interval = 10
    dx = (expected[1, 0, 0] - expected[0, 0, 0]) / num_interval
    dy = (expected[1, 0, 1] - expected[0, 0, 1]) / num_interval
    extra_points = [
        [[expected[0, 0, 0] + i * dx, expected[0, 0, 1] + i * dy]]
        for i in range(1, num_interval)
    ]
    contour = np.append(contour, extra_points, axis=0)

    approx = approximate_contour(contour.astype(expected.dtype))

    assert expected.shape == approx.shape, "Incorrect output shape"
    assert all(point in expected for point in approx), "Incorrect output points"


def test_get_edged_image():
    image_edged = get_edged_image(IMAGE, (3, 3), (0, 50))

    assert isinstance(image_edged, np.ndarray), "Wrong return type"
    assert image_edged.dtype == np.uint8, "Wrong data type"


def test_get_sorted_contour():
    image = cv2.resize(IMAGE, (1500, 1500))
    image_edged = get_edged_image(image, (5, 5), (0, 50))
    contours = get_sorted_contours(image_edged)

    assert isinstance(contours, list), "Wrong return type"
    assert all(
        isinstance(contour, np.ndarray) for contour in contours
    ), "Wrong element type"
    assert all(contour.dtype == np.int32 for contour in contours), "Wrong element type"
    for i in range(1, len(contours)):
        assert cv2.contourArea(contours[i - 1]) >= cv2.contourArea(
            contours[i]
        ), "List not sorted in descending order by area"
