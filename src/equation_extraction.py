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
        # convert to grayscale and blur to smooth
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image_edged = cv2.Canny(image, 0, 50)
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        contours, _ = cv2.findContours(
            image_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break
        approx = rectify(target)
        points = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

        M = cv2.getPerspectiveTransform(approx, points)
        document = cv2.warpPerspective(self.annotated_image, M, (800, 800))

        cv2.drawContours(self.annotated_image, [target], -1, (0, 255, 0), 2)
        # document = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)
        return document

    def extract_equations(self, path):
        document = self.extract_document(path)
        self.annotated_document = document.copy()
        # convert to grayscale and blur to smooth
        document = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        document = cv2.GaussianBlur(document, (3, 3), 0)
        document_edged = cv2.Canny(document, 0, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        document_edged = cv2.dilate(document_edged, kernel, iterations=1)

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        contours, _ = cv2.findContours(
            document_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        targets = []
        # get approximate contour
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                if cv2.contourArea(approx) >= 30000:
                    if not targets:
                        targets.append(approx)
                    elif not any(
                        has_intersection(target, approx) for target in targets
                    ):
                        targets.append(approx)
                if len(targets) == 5:
                    break
        targets = sorted(targets, key=lambda c: c[0][0][1])
        equations = []
        for target in targets:
            approx = rectify(target)
            points = np.float32([[0, 0], [800, 0], [800, 100], [0, 100]])

            M = cv2.getPerspectiveTransform(approx, points)
            equations.append(
                cv2.warpPerspective(self.annotated_document, M, (800, 100))
            )

            cv2.drawContours(self.annotated_document, [target], -1, (0, 255, 0), 2)
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
