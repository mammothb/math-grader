"""A collection of geometric image transformation utility functions
"""
import cv2
import numpy as np


def convert_to_rectangle(contour):
    """Return the bounding rectangle for the provided contour. The
    returned coordinates follows the ordering in rectify():
    [lower left, lower right, upper right, upper left]

    Args:
        contour (np.ndarray of int): Coordinates of a contour

    Returns:
        np.ndarray of int: The bounding rectangle of the input contour
    """
    rect = cv2.boundingRect(contour)
    return np.array(
        [
            [[rect[0], rect[1]]],
            [[rect[0] + rect[2], rect[1]]],
            [[rect[0] + rect[2], rect[1] + rect[3]]],
            [[rect[0], rect[1] + rect[3]]],
        ]
    )


def get_perspective_transform(contour, x, y):
    """Calculates the perspective transform from the approximated
    contour to the desired rectangle

    Args:
        contour (np.ndarray of float): Coordinates of the approximated
            contour
        x (int): Width of the desired rectangle
        y (int): Height of the desired rectangle

    Returns:
        np.ndarray of float: A 3 x 3 matrix of the perspective
            transform
    """
    # reorders the contour coordinates so that it matches the order of
    # the destination coordinates
    rect = rectify(contour)
    points = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

    return cv2.getPerspectiveTransform(rect, points)


def has_intersection(contour_ref, contour_query):
    """Check if there is any intesection before the two provided
    contours

    Args:
        contour_ref (np.ndarray of float): The reference contour
        contour_query (np.ndarray of float): The contour to check

    Returns:
        bool: True if there is any intersection between the two
            contours. False if there is no intersection
    """
    # Loop through all points in the contour
    for point in contour_query:
        # find point that intersect the ref contour
        if cv2.pointPolygonTest(contour_ref, tuple(point[0]), True) >= 0:
            return True

    return False


def rectify(contour):
    """Reorders the coordinates so that it follows
    [lower left, lower right, upper right, upper left]

    Args:
        h (np.ndarray of float): Input contours which describe a 4
        sided polygon

    Returns:
        np.ndarray of float: The reordered contours
    """
    contour = contour.reshape((4, 2))
    contour_reordered = np.zeros((4, 2), dtype=np.float32)

    add = np.sum(contour, axis=1)
    contour_reordered[0] = contour[np.argmin(add)]
    contour_reordered[2] = contour[np.argmax(add)]

    diff = np.diff(contour, axis=1)
    contour_reordered[1] = contour[np.argmin(diff)]
    contour_reordered[3] = contour[np.argmax(diff)]

    return contour_reordered

def trim_and_otsu_threshold(image, trim_proportion=0.08):
    height, width = image.shape
    trim = int(min(trim_proportion * height, trim_proportion * width))
    image = image[trim : height - 2 * trim, trim : width - 2 * trim]
    out = cv2.GaussianBlur(image, (7, 7), 0)
    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out
