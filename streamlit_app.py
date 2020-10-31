import os.path
from pathlib import Path
import urllib

import numpy as np
from PIL import Image
import streamlit as st

from src.bounding_box import get_bounding_box
from src.equation_extraction import EquationExtractor
from src.inference import mark_equations, parse_equation
from src.utils.geometric import trim_and_otsu_threshold
from src.utils.visualize import labeled_annotate_image


st.set_page_config(
    page_title="Math Grader",
    page_icon=":pencil:",
    layout="centered",
    initial_sidebar_state="auto",
)
# Remove Deprecation Warning
st.set_option("deprecation.showfileUploaderEncoding", False)


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("INSTRUCTIONS.md"))

    # Download external dependencies.
    for filename in DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Show instructions", "Run the app"],
    )
    if app_mode == "Show instructions":
        st.sidebar.success("To continue select 'Run the app'.")
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()


# This file downloader demonstrates Streamlit animation.
def download_file(filename):
    dst_dir = Path.cwd() / "dependency"
    dst_dir.mkdir(parents=True, exist_ok=True)
    file_path = dst_dir / filename
    # Don't download the file twice. (If possible, verify the download
    # using the file length.)
    if file_path.exists():
        if "size" not in DEPENDENCIES[filename]:
            return
        elif file_path.stat().st_size == DEPENDENCIES[filename]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning(f"Downloading {file_path}...")
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(DEPENDENCIES[filename]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        f"Downloading {str(file_path)}... "
                        f"({counter / MEGABYTES:6.2f}/{length / MEGABYTES:6.2f} MB)"
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def run_the_app():
    st.header("Demo")
    st.markdown("**1) Upload a math paper**")
    upload_file = st.file_uploader(
        "Upload a math worksheet (.jpg/.png)", type=["jpg", "png"], key="demo"
    )
    if upload_file is not None:
        uploaded_image = Image.open(upload_file)
        st.image(
            uploaded_image,
            caption="Uploaded image.",
            channels="RGB",
            use_column_width=True,
        )
        tmp_dir = Path.cwd() / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        image_path = tmp_dir / "worksheet.jpg"
        with open(image_path, "wb") as outfile:
            outfile.write(upload_file.getbuffer())
    else:
        st.warning("Please upload an image.")
        st.stop()

    st.markdown("**2) Let's run the model!**")

    def end_to_end(image_path):
        equation_extractor = EquationExtractor()
        equations, targets = equation_extractor.extract_equations(image_path)
        equations = [
            trim_and_otsu_threshold(np.array(Image.fromarray(equation).convert("L")))
            for equation in equations
        ]

        equation_boxes = []
        for equation in equations:
            boxes = get_bounding_box(equation)
            equation_boxes.append(boxes)

        crop_equation_images = []
        for image, boxes in zip(equations, equation_boxes):
            image_arr = np.array(image)
            crop_images = [
                image_arr[top : top + height, left : left + width]
                for left, top, width, height in boxes
            ]
            crop_equation_images.append(crop_images)

        image_paths = []
        for i, equation_images in enumerate(crop_equation_images):
            image_paths.append([])
            for j, image in enumerate(equation_images):
                image_path = tmp_dir / f"eqn_{i}_char_{j}.jpg"
                Image.fromarray(image, mode="L").save(image_path)
                image_paths[-1].append(image_path)

        equations = []
        for char_paths in image_paths:
            equation = parse_equation(char_paths)
            equations.append(equation)

        marked_equations = mark_equations(equations)

        document_image = equation_extractor.unannotated_document.copy()
        labeled_annotate_image(document_image, targets, marked_equations)

        return document_image

    marked_document = None
    with st.spinner("Wait for it..."):
        marked_document = end_to_end(image_path)
        st.success("Ran successfully")
    if marked_document is not None:
        st.image(marked_document)


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = f"https://raw.githubusercontent.com/mammothb/math-grader/master/{path}"
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


DEPENDENCIES = {
    "model_weights.h5": {
        "url": "https://github.com/mammothb/math-grader/raw/master/model/mnist_symbols_imbalance_993.h5",
        "size": 10728912,
    },
}


if __name__ == "__main__":
    main()
