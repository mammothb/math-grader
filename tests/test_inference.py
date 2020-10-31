from pathlib import Path
import urllib

import pytest

from src.inference import (
    classify_image,
    convert_image_to_tensor,
    mark_equations,
    parse_equation,
    resize_image,
)

DST_DIR = Path.cwd() / "dependency"
DST_DIR.mkdir(parents=True, exist_ok=True)
with open(DST_DIR / "model_weights.h5", "wb") as output_file:
    with urllib.request.urlopen(
        "https://github.com/mammothb/math-grader/"
        "raw/master/model/mnist_symbols_imbalance_993.h5"
    ) as response:
        MEGABYTES = 2.0 ** 20.0
        while True:
            data = response.read(8192)
            if not data:
                break
            output_file.write(data)


def test_resize_image():
    image_path = Path.cwd() / "data" / "example" / "-" / "-_example.jpg"

    expected_dim = 28
    image = resize_image(image_path)

    assert image.height == expected_dim, "Wrong image height"
    assert image.width == expected_dim, "Wrong image height"


def test_resize_image_invalid_path():
    image_path = Path.cwd() / "invalid.jpg"
    with pytest.raises(ValueError) as excinfo:
        _ = resize_image(image_path)
    assert str(excinfo.value) == "Invalid image path", "Failed to raise error"


def test_convert_image_to_tensor():
    image_path = Path.cwd() / "data" / "example" / "-" / "-_example.jpg"

    image = resize_image(image_path)
    tensor = convert_image_to_tensor(image)
    assert tensor.shape == (1, 28, 28, 1), "Wrong tensor shape"


def test_classify_image():
    image_path = Path.cwd() / "data" / "example" / "-" / "-_example.jpg"

    prediction = classify_image(image_path)
    assert prediction[0] == "-", "Wrong prediction"


def test_parse_equation():
    image_paths = [
        Path.cwd() / "data" / "example" / "5" / "5_example.jpg",
        Path.cwd() / "data" / "example" / "-" / "-_example.jpg",
        Path.cwd() / "data" / "example" / "3" / "3_example.jpg",
        Path.cwd() / "data" / "cropped" / "=_cropped.jpg",
        Path.cwd() / "data" / "example" / "2" / "2_example.jpg",
    ]
    prediction = parse_equation(image_paths)

    assert prediction == "5-3=2"


def test_mark_equations():
    correct_equations = ["2+3=5", "3*5=15", "4-3=1"]
    assert all(mark_equations(correct_equations)), "Wrong mark equations"

    wrong_equations = ["2+2=5", "1+3", "1/0=0"]
    assert not any(mark_equations(wrong_equations)), "Failed to mark wrong equations"

    forbidden_equations = ["print('hello world')=0"]
    assert not any(
        mark_equations(forbidden_equations)
    ), "Failed to mark forbidden equations"
