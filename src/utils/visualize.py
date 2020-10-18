"""A collection of visualization utility functions
"""
import cv2


def annotate_image(image, contour):
    """Draws the contour on the provided image

    Args:
        image (np.ndarray of uint8): The input image
        contour (np.ndarray of float): Coordinates of the contour
    """
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
