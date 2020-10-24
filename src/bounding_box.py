from itertools import permutations
import math

import cv2
import numpy as np


def default_preprocessor(threshold=0.875, invert=True, n_dilation_iter=2):
    """Default image preprocessing function to enhance the bounding box
    detection performance.

    Args:
        threshold (float, optional): The threshold value used to
            convert the input image to binary. Defaults to 0.875.
        invert (bool, optional): Flag to determine whether to use
            THRESH_BINARY_INV or THRESH_BINARY. Defaults to True.
        n_dilation_iter (int, optional): Number of iterations to
            perform the dilation operation. Defaults to 2.
    """

    def processor_func(image):
        _, out = cv2.threshold(
            image,
            math.floor(threshold * 255),
            255,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if n_dilation_iter > 0:
            out = cv2.dilate(out, kernel, iterations=n_dilation_iter)
        return out

    return processor_func


def default_postprocessor(threshold=0.5):
    """Default post processing function to remove overlapped bounding
    boxes

    Args:
        threshold (float, optional): Threshold of the amount of
            horizontal overlap for the two bounding boxes to be
            considered as "vertically stacked". Defaults to 0.5.
    """

    def processor_func(boxes):
        old_boxes = boxes
        new_boxes = merge_y_stacked_bounding_boxes(boxes, threshold)
        while len(new_boxes) != len(old_boxes):
            old_boxes = new_boxes
            new_boxes = merge_y_stacked_bounding_boxes(new_boxes, threshold)
        return new_boxes

    return processor_func


def non_max_suppression(boxes, threshold):
    """An optimized implementation of the non-maximum suppression
    algorithm.
    Adapted from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        boxes (np.ndarray of int): Array of bounding box coordinates
        threshold (float): Threshold of area overlap

    Returns:
        list of np.ndarray of int: List of bounding box coordinates
    """
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort the bounding boxes by the lower-right y-coordinate
    idxs = np.argsort(y2)
    # keep looping while some indices still remain in the indices list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest coordinates for the start of the bounding
        # box and the smallest coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        overlap = (width * height) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > threshold)[0]))
        )

    return list(boxes[pick].astype("int"))


def merge_boxes(box1, box2):
    """Merge boxes by creating the smallest box that can bound both
    input boxes

    Args:
        box1 (np.ndarray of int): The first input box
        box2 (np.ndarray of int): The second input box

    Returns:
        tuple of int: The [left, top, width, height] of the merged box
    """
    left = min(box1[0], box2[0])
    right = max(box1[0] + box1[2], box2[0] + box2[2])
    top = min(box1[1], box2[1])
    bottom = max(box1[1] + box1[3], box2[1] + box1[3])
    return left, top, right - left, bottom - top


def merge_y_stacked_bounding_boxes(boxes, threshold):
    """Merge vertically stacked boxes. Multiple bounding boxes will be
    detected for symbols such as "=" and "div", this function will
    create a merged (larger) bounding box which will bound the whole
    symbol.

    Args:
        boxes (list of np.ndarray of int): List of bounding boxes
        threshold (float): Threshold for horizontal overlap for the
            boxes to be considered to be "vertically stacked"

    Returns:
        list of np.ndarray of int: List of bounding box coordinates
    """
    merge_pairs = set()
    for idx1, idx2 in permutations(range(len(boxes)), 2):
        box1, box2 = boxes[idx1], boxes[idx2]
        l1, r1, l2, r2 = box1[0], box1[0] + box1[2], box2[0], box2[0] + box2[2]
        if l1 < l2 and r1 < r2 and box1[2] and (r1 - l2) / box1[2] > threshold:
            merge_pairs.add(frozenset((idx1, idx2)))
        elif l1 <= l2 and r1 >= r2:
            merge_pairs.add(frozenset((idx1, idx2)))
    merged_boxes = [merge_boxes(boxes[idx1], boxes[idx2]) for idx1, idx2 in merge_pairs]
    merged_idxs = set(box for pair in merge_pairs for box in pair)
    merged_boxes.extend(box for idx, box in enumerate(boxes) if idx not in merged_idxs)
    return non_max_suppression(np.array(merged_boxes), threshold)


def get_bounding_box(
    image, preprocessor=default_preprocessor(), postprocessor=default_postprocessor()
):
    """Get per character bounding boxes from the given image

    Args:
        image (np.ndarray of int): The input image
        preprocessor (function, optional): The preprocessing function
            to be applied to the input image. Defaults to
            default_preprocessor().
        postprocessor (function, optional): The postprocessing function
            to be applied on the generated bounding boxes. Defaults to
            default_postprocessor().

    Returns:
        list of np.ndarray of int: A list of the bounding boxes
            detected from the image
    """
    if preprocessor:
        image = preprocessor(image)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Check hierarchy parent is -1 to ensure that this bounding box is not
    # contained in another box
    boxes = [
        cv2.boundingRect(cnt)
        for cnt, (_, _, _, parent) in zip(contours, hierarchy[0])
        if parent == -1
    ]
    if postprocessor:
        boxes = postprocessor(boxes)
    boxes.sort(key=lambda x: x[0])
    return boxes
