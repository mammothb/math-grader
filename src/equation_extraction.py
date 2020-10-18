"""This module contains the EquationExtract class and helper functions
necessary to extract equation boxes from a photo/scan of a worksheet
"""
import cv2

from src.utils.geometric import (
    convert_to_rectangle,
    get_perspective_transform,
    has_intersection,
    rectify,
)
from src.utils.structural import (
    approximate_contour,
    get_edged_image,
    get_sorted_contours,
)
from src.utils.visualize import annotate_image


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
