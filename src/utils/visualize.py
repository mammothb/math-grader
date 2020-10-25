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

def labeled_annotate_image(image, contours, labels):
    for contour, label in zip(contours, labels):
        if label:
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        else:
            cv2.drawContours(image, [contour], 0, (255, 0, 0), 2)
