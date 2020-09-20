import cv2
import numpy as np


class EquationExtractor:
    def __init__(self):
        self.annotated_image = None
        self.annotated_document = None

    def extract_document(self, path):
        image = cv2.imread(path)
        # resize image so it can be processed
        # choose optimal dimensions such that important content is not lost
        image = cv2.resize(image, (1500, 1500))
        self.annotated_image = image.copy()
        image_edged = get_edged_image(image, (5, 5), (0, 50))
        contours = get_sorted_contours(image_edged)
        for c in contours:
            approx = approximate_contour(c)
            if len(approx) == 4:
                target = approx
                break

        M = get_perspective_transform(target, 800, 800)
        document = cv2.warpPerspective(self.annotated_image, M, (800, 800))
        annotate_image(self.annotated_image, target)

        return document

    def extract_equations(self, path):
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
        for c in contours:
            approx = approximate_contour(c)
            if len(approx) == 4 and cv2.contourArea(approx) >= 30000:
                if not targets:
                    targets.append(approx)
                elif not any(has_intersection(target, approx) for target in targets):
                    targets.append(approx)
            if len(targets) == 5:
                break
        # Sort contours by y-coordinate
        targets = sorted(targets, key=lambda c: c[0][0][1])
        equations = []
        for target in targets:
            M = get_perspective_transform(target, 800, 100)
            equations.append(
                cv2.warpPerspective(self.annotated_document, M, (800, 100))
            )
            annotate_image(self.annotated_document, target)

        return equations


def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def has_intersection(contour_ref, contour_query):
    intersecting_points = []
    # Loop through all points in the contour
    for point in contour_query:
        # find point that intersect the ref contour
        if cv2.pointPolygonTest(contour_ref, tuple(point[0]), True) >= 0:
            intersecting_points.append(point[0])

    return len(intersecting_points) > 0


def get_edged_image(image, kernel_size, canny_threshold):
    # convert to grayscale and blur to smooth
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, kernel_size, 0)
    image_edged = cv2.Canny(image, *canny_threshold)
    return image_edged


def get_sorted_contours(image):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def approximate_contour(contour):
    return cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)


def get_perspective_transform(contour, x, y):
    approx = rectify(contour)
    points = np.float32([[0, 0], [x, 0], [x, y], [0, y]])

    return cv2.getPerspectiveTransform(approx, points)


def annotate_image(image, target):
    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
