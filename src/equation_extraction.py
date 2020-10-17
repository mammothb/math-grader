"""This module contains the EquationExtract class and helper functions
necessary to extract equation boxes from a photo/scan of a worksheet
"""
import cv2
import numpy as np


class EquationExtractor:
    """This class extracts and transform the worksheet from a photo or
    a scan and performs extraction of the equation boxes.

    Attributes:
        annotated_image (np.ndarray of uint8): A transformed image of
            the photo/scans with bounding box drawn on the detected
            worksheet
        annotated_document (np.ndarray of uint8): An image of the
            extracted worksheet with bounding boxes drawn on each of
            the detected equation boxes
    """

    min_equation_box_area = 30000
    num_equations = 5
    num_rectangle_vertices = 4

    def __init__(self):
        self.annotated_image = None
        self.annotated_document = None

    def extract_document(self, path):
        """Extract the worksheet from a photo/scan containing the
        worksheet. The extracted worksheet will be rescaled to a square
        shape (1500x1500).

        Note: This method assumes that the image provided contains the
        worksheet with all four edges visible.

        Args:
            path (pathlib.Path): Path to the image of the photo/scan
                containing the worksheet

        Returns:
            np.ndarray of uint8: A transformed version of the extracted
                worksheet
        """
        image = cv2.imread(str(path))
        # resize image so it can be processed
        # choose optimal dimensions such that important content is not lost
        image = cv2.resize(image, (1500, 1500))
        self.annotated_image = image.copy()
        image_edged = get_edged_image(image, (5, 5), (0, 50))
        contours = get_sorted_contours(image_edged)
        for contour in contours:
            approx = approximate_contour(contour)
            # If the approximated polygon has 4 vertices, we take it to
            # be the worksheet and return early
            if len(approx) == self.num_rectangle_vertices:
                target = approx
                break

        transformation_matrix = get_perspective_transform(target, 800, 800)
        document = cv2.warpPerspective(
            self.annotated_image, transformation_matrix, (800, 800)
        )
        annotate_image(self.annotated_image, target)

        return document

    def extract_equations(self, path):
        """Extract equation boxes from the photo/scan of the worksheet

        Args:
            path (pathlib.Path): Path to the image of the photo/scan
                containing the worksheet

        Returns:
            list of np.ndarray of uint8: A list of the extracted
                equation boxes
        """
        document = self.extract_document(path)
        self.annotated_document = document.copy()
        document_edged = get_edged_image(document, (3, 3), (0, 100))

        # dilate images to thicken the equation box lines for easier
        # detection
        document_edged = cv2.dilate(
            document_edged,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours = get_sorted_contours(document_edged)
        targets = []
        # get approximate contour
        for contour in contours:
            approx = approximate_contour(contour)
            # Only select 4 sided contours above a certain area
            if (
                len(approx) == self.num_rectangle_vertices
                and cv2.contourArea(approx) >= self.min_equation_box_area
            ):
                if not targets:
                    targets.append(approx)
                elif not any(has_intersection(target, approx) for target in targets):
                    targets.append(approx)
            if len(targets) == self.num_equations:
                break
        # Sort contours by y-coordinate
        targets = sorted(targets, key=lambda c: c[0][0][1])
        equations = []
        for target in targets:
            target = convert_to_rectangle(target)
            transformation_matrix = get_perspective_transform(target, 800, 100)
            equations.append(
                cv2.warpPerspective(
                    self.annotated_document, transformation_matrix, (800, 100)
                )
            )
            annotate_image(self.annotated_document, target)

        return equations


def rectify(contour):
    """Reorders the coordinates so that it follows
    [upper left, lower left, lowr right, upper right]

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


def convert_to_rectangle(contour):
    """Return the bounding rectangle for the provided contour

    Args:
        contour (np.ndarray of int): Coordinates of a contour

    Returns:
        np.ndarray of int: The bounding rectangle of the input contour
    """
    rect = cv2.boundingRect(contour)
    return np.array(
        [
            [[rect[0] + rect[2], rect[1]]],
            [[rect[0] + rect[2], rect[1] + rect[3]]],
            [[rect[0], rect[1] + rect[3]]],
            [[rect[0], rect[1]]],
        ]
    )


def annotate_image(image, contour):
    """Draws the contour on the provided image

    Args:
        image (np.ndarray of uint8): The input image
        contour (np.ndarray of float): Coordinates of the contour
    """
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
