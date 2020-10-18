import cv2
import numpy as np

from src.utils.geometric import (
    convert_to_rectangle,
    get_perspective_transform,
    has_intersection,
    rectify,
)


CONTOUR = np.array([[[3, 2]], [[11, 4]], [[4, 21]], [[10, 15]]])


def test_convert_to_rectangle():
    rectangle = convert_to_rectangle(CONTOUR)

    # Check that the output rectangle fully encloses the input contour
    x_min = np.min(rectangle[:, 0, 0])
    x_max = np.max(rectangle[:, 0, 0])
    y_min = np.min(rectangle[:, 0, 1])
    y_max = np.max(rectangle[:, 0, 1])

    assert x_min <= np.min(CONTOUR[:, 0, 0]), "x_min too large"
    assert x_max >= np.max(CONTOUR[:, 0, 0]), "x_max too small"
    assert y_min <= np.min(CONTOUR[:, 0, 1]), "y_min too large"
    assert y_max >= np.max(CONTOUR[:, 0, 1]), "y_max too small"

    # Check that the output is a rectangle assuming the points are
    # ordered: [lower left, lower right, upper right, upper left]
    assert rectangle[0, 0, 1] == rectangle[1, 0, 1], "Lower line not horizontal"
    assert rectangle[2, 0, 1] == rectangle[3, 0, 1], "Upper line not horizontal"
    assert rectangle[0, 0, 0] == rectangle[3, 0, 0], "Left line not vertical"
    assert rectangle[1, 0, 0] == rectangle[2, 0, 0], "Right line not vertical"


def test_rectify():
    # Start with a contour order clockwise starting from the upper left
    # corner
    contour = np.array([[[3, 11]], [[14, 12]], [[15, 4]], [[2, 4]]])
    contour = rectify(contour)
    expected = np.array([[2, 4], [15, 4], [14, 12], [3, 11]], dtype=np.float32)
    assert (expected == contour).all(), "Wrong ordering of points"


def test_get_perspective_transform():
    x = 12
    y = 34

    transform_matrix = get_perspective_transform(CONTOUR, x, y)
    expected = cv2.getPerspectiveTransform(
        rectify(CONTOUR),
        np.float32([[0, 0], [x, 0], [x, y], [0, y]]),
    )
    assert (expected == transform_matrix).all(), "Transform matrix doesn't match"


def test_has_intersection():
    contour_query_1 = np.array([[[4, 3]], [[11, 4]], [[4, 21]], [[10, 15]]])
    assert has_intersection(
        CONTOUR, contour_query_1
    ), "Inaccurate intersection detection"

    # Shift the contour_query to the right of the reference contour
    contour_query_2 = np.array([[[13, 2]], [[21, 4]], [[14, 21]], [[20, 15]]])
    assert not has_intersection(
        CONTOUR, contour_query_2
    ), "Inaccurate intersection detection"
