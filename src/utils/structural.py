"""A collection of image filtering utility functions
"""
import cv2


def approximate_contour(contour):
    """Approximates the provided contour with a simplified polygon so
    that the distance between the contour and the simplified polygon is
    less then 2% of the contour perimeter

    Args:
        contour (np.ndarray of int): The contour to be approximated

    Returns:
        np.ndarray of int: The approximated polygon
    """
    return cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)


def get_edged_image(image, kernel_size, canny_threshold):
    """Perform canny edge detection on the provided image. Gaussian
    blur is applied before performing canny edge detection.

    Args:
        image (np.ndarray of uint8): The image to perform edge
            detection on
        kernel_size (tuple of int): Kernel size for the gaussian blur
            step
        canny_threshold (tuple of float): Lower and upper threshold for
            the hysteresis procedure

    Returns:
        np.ndarray of uint8: The binary edge map
    """
    # convert to grayscale and blur to smooth
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, kernel_size, 0)
    image_edged = cv2.Canny(image, *canny_threshold)
    return image_edged


def get_sorted_contours(image):
    """Returns a sorted list of contours from the provided image. The
    list is sort in descending order based on contour area.

    Args:
        image (np.ndarray of uint8): The provided image to find
            contours from

    Returns:
        list of np.ndarray of int: A list of numpy arrays of contour
            coordinates sorted based on contour area in descending
            order
    """
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours
