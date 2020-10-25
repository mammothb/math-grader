import ast
import logging
import os
import os.path
from pathlib import Path

from PIL import Image, ImageOps
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("inference")

CHARS = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "/", "*"]


def resize_image(img_path):
    """Resize and invert image, compliant with model's requirement

    Args:
        img_path (str): Path to the image file

    Returns:
        PIL.Image: The resized image
    """
    if not os.path.exists(img_path):
        raise ValueError("Invalid image path")

    img_dim = 28
    img = Image.open(img_path)
    width, height = img.size
    if width != height:
        big_side = width if width > height else height
        background = Image.new("L", (big_side, big_side), (255,))
        offset = (
            int(round((big_side - width) / 2, 0)),
            int(round((big_side - height) / 2, 0)),
        )
        background.paste(img, offset)
    img = ImageOps.invert(img.resize((img_dim, img_dim)))

    return img


def convert_image_to_tensor(img):
    """Convert image to tensor, compliant with model's requirement

    Args:
        img (PIL.Image): The image to be converted

    Returns:
        tf.Tensor: The converted tensor
    """
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255
    # Create a batch
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model",
        "mnist_symbols_imbalance_993.h5",
    )
    model = tf.keras.models.load_model(model_path)
    LOG.info("Loaded model")
    return model


def classify_image(img_path, model=None):
    """Loads the digit/symbol classification model and classfies the
    characters based on the provided path

    Args:
        img_path (str): Path to the image file
        model (tf.Model, optional): The digit/symbol classification
            model. Defaults to None.

    Returns:
        str: The predicted food category
    """
    img_array = convert_image_to_tensor(resize_image(img_path))

    if model is None:
        model = load_model()

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    idx = tf.math.argmax(score)

    return CHARS[idx], score[idx].numpy()


def parse_equation(img_paths, model=None):
    """Loads the digit/symbol classification model and parses the
    segmented equation images

    Args:
        img_paths (list of str): List of paths to the segmented
            equation image files
        model (tf.Model, optional): The digit/symbol classification
            model. Defaults to None.

    Returns:
        str: The predicted equation
    """
    if model is None:
        model = load_model()

    equation = "".join(classify_image(img_path, model)[0] for img_path in img_paths)

    return equation


def is_valid_ast_tree(ast_tree):
    """Check that all nodes in the AST tree is in the whitelisted list
    of ast grammar

    Args:
        ast_tree (ast.Expression): The parsed AST tree

    Returns:
        bool: True if all nodes in the tree are whitelisted, False if
            there is a non-whitelisted node
    """
    whitelist = (ast.Expression, ast.BinOp, ast.operator, ast.Num)
    return all(isinstance(node, whitelist) for node in ast.walk(ast_tree))


def eval_ast(ast_tree):
    """Evaluates the validated AST tree

    Args:
        ast_tree (ast.Expression): The parsed AST tree

    Returns:
        int: The evaluation result. Based on the intended use case,
            this should return an int.
    """
    return eval(compile(ast_tree, filename="", mode="eval"), {"__builtins__": None})


def mark_equations(equations):
    """Marks whether the equations are correct

    Args:
        equations (list of str): List of strings of the parsed
            equations

    Returns:
        list of bool: List of where the equations are correct (both
            mathematically and syntactically)
    """
    answers = []
    for equation in equations:
        try:
            LOG.info("Parsed equation: %s", equation)
            lhs, rhs = equation.split("=")
            l_tree = ast.parse(lhs, mode="eval")
            r_tree = ast.parse(rhs, mode="eval")
            if is_valid_ast_tree(l_tree) and is_valid_ast_tree(r_tree):
                answers.append(eval_ast(l_tree) == eval_ast(r_tree))
            else:
                raise SyntaxError
        except (SyntaxError, ValueError, ZeroDivisionError):
            answers.append(False)
    return answers
