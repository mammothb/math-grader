import logging
import os
import os.path

from PIL import Image, ImageOps
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

CHARS = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "/", "*"]


def resize_image(img_path):
    """
    Resize and invert image, compliant with model's requirement

    Parameters
    ----------
    img_path : str
        Path to the image file

    Returns
    -------
    PIL.Image
        The resized image
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
    """
    Convert image to tensor, compliant with model's requirement

    Parameters
    ----------
    img : PIL.Image
        The image to be converted

    Returns
    -------
    tf.Tensor
        The converted tensor
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
    print(model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info("Loaded model")
    return model


def classify_image(img_path, model=None):
    """
    Loads the food classification model and classfies the food based
    on the provided path

    Parameters
    ----------
    img_path : str
        Path to the image file

    Returns
    -------
    str
        The predicted food category
    """
    img_array = convert_image_to_tensor(resize_image(img_path))

    if model is None:
        model = load_model()

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    idx = tf.math.argmax(score)

    return CHARS[idx], score[idx].numpy()


def parse_equation(img_paths, model=None):
    """
    Loads the food classification model and parses the segmented
    equation images

    Parameters
    ----------
    img_paths : list of str
        List of paths to the segmented equation image files

    Returns
    -------
    str
        The predicted equation
    """
    if model is None:
        model = load_model()

    equation = "".join(classify_image(img_path, model)[0] for img_path in img_paths)

    return equation


def mark_equations(equations):
    """
    Marks whethere the equations are correct

    Parameters
    ----------
    equations : list of str
        List of strings of the parsed equations

    Returns
    -------
    list of bool
        List of where the equations are correct (both mathematically
        and syntactically)
    """
    answers = []
    for equation in equations:
        try:
            lhs, rhs = equation.split("=")
            answers.append(eval(lhs) == eval(rhs))
        except (SyntaxError, ValueError, ZeroDivisionError):
            answers.append(False)
    return answers
