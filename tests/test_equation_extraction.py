from pathlib import Path

import numpy as np

from src.equation_extraction import EquationExtractor

IMAGE_PATH = Path.cwd() / "data" / "example_filled.jpg"


def test_equation_extractor_constructor():
    extractor = EquationExtractor()

    assert extractor.annotated_image is None, "annotated_image not declared"
    assert extractor.annotated_document is None, "annotated_document not declared"


def test_equation_extractor_extract_document():
    extractor = EquationExtractor()
    document = extractor.extract_document(IMAGE_PATH)

    assert isinstance(document, np.ndarray), "Wrong return type"
    assert document.dtype == np.uint8, "Wrong return data type"
    assert document.shape == (800, 800, 3), "Wrong return shape"

    assert isinstance(
        extractor.annotated_image, np.ndarray
    ), "Incorrect annotated_image type"
    assert (
        extractor.annotated_image.dtype == np.uint8
    ), "Incorrect annotated_image data type"
    assert extractor.annotated_image.shape == (
        1500,
        1500,
        3,
    ), "Incorrect annotated_image shape"


def test_equation_extractor_extract_equations():
    extractor = EquationExtractor()
    equations = extractor.extract_equations(IMAGE_PATH)

    assert isinstance(equations, list), "Wrong return type"
    assert all(
        isinstance(equation, np.ndarray) for equation in equations
    ), "Wrong return element type"
    assert all(
        equation.dtype == np.uint8 for equation in equations
    ), "Wrong return element data type"
    assert all(
        equation.shape == (100, 800, 3) for equation in equations
    ), "Wrong return element shape"

    assert isinstance(
        extractor.annotated_document, np.ndarray
    ), "Wrong annotated_document type"
    assert (
        extractor.annotated_document.dtype == np.uint8
    ), "Wrong annotated_document data type"
    assert extractor.annotated_document.shape == (
        800,
        800,
        3,
    ), "Wrong annotated_document shape"
