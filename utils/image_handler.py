import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Tuple

import cv2 as cv
import numpy as np
import requests
from cv2.typing import MatLike
from PIL import Image


class picture:
    def __init__(self, source: str) -> None:
        if isinstance(source, str):
            if source.startswith(("https://", "http://")):
                self.IMG_TYPE = "url"
            else:
                self.IMG_TYPE = "file_path"
        else:
            return
        if self.IMG_TYPE == "file_path":
            self.img = np.array(cv.imread(source))
            if self.img is None:
                raise FileNotFoundError(f"No image found at {source}")

        elif self.IMG_TYPE == "url":
            response = requests.get(source)
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.img = np.array(image)
        else:
            raise Exception(f"Failed to read image from {source}")

    def to_gray(self) -> None:
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def to_bgr(self) -> None:
        self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)

    def resize(self, scale: float = 0.75) -> None:
        width = int(self.img.shape[1] * scale)
        height = int(self.img.shape[0] * scale)

        dimensions = (width, height)

        cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)

    #   TODO    draw a rectangle around the face
    def draw_face_rect(self):
        pass

    #   TODO    resize the image to a semi-standar size, i.e constant width or height with the same aspect ratio
    def semi_std_size(self) -> None:
        pass


def read_image(img_input: str, gray=False) -> MatLike | Tuple[MatLike, MatLike]:
    """
    Reads an image from a url, path or numpy object
    Input: string or numpy object
    Outputs:
    rgb_img: RGB image
    gray_img: Grayscale image
    """
    if isinstance(img_input, str):
        if img_input.startswith(("https://", "http://")):
            response = requests.get(img_input)
            image_bytes = response.content
            img = Image.open(BytesIO(image_bytes))
            if gray is True:
                gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
                return rgb_img, gray_img
            else:
                return rgb_img
        else:
            bgr_img = cv.imread(img_input)
            if bgr_img is None:
                raise FileNotFoundError(f"No image found at {img_input}")
    # If the input is a numpy array, assume it's an image
    elif isinstance(img_input, np.ndarray):
        bgr_img = img_input
    else:
        print(img_input)
        raise ValueError(
            "Input should be an image path (str) or an image object (numpy.ndarray)"
        )

    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

    if gray is True:
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        return rgb_img, gray_img
    else:
        return rgb_img


def create(new_img_path: str, img: np.ndarray):
    cv.imwrite(f"{new_img_path}.jpg", img)


def remove(img_path: str):
    try:
        os.remove(img_path)
    except FileNotFoundError:
        print("An error occurred trying to delete!" + f"{img_path}.jpg")
    return


"""
Takes an image as a numpy ndarray, a new image path, and the coordinates of the crop
"""


def crop_imgs(img: np.ndarray, new_img_path: str, imgs_coord: list):
    bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for i, (x1, y1, x2, y2) in enumerate(imgs_coord):
        clip = bgr_img[y1:y2, x1:x2]
        create(f"{new_img_path}_{i+1}", clip)


def draw_borders(img: np.ndarray, imgs_coord: list):
    for x1, y1, x2, y2 in imgs_coord:
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def plot(img: np.ndarray, title: str):
    try:
        cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow(f"{title}", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except cv.error:
        print("No img was found in the given path")
    return


def points2rotation_format(fl):
    fl = np.array(fl)
    face_locations = fl[:, [1, 2, 3, 0]]
    return face_locations


class logger:
    def __init__(self):
        logging.basicConfig(filename="log.log", level=logging.INFO)
        self.current_time = f"{datetime.now().replace(microsecond=0)}"

    def add(self, message: str):
        logging.info(f"{self.current_time}: {message}")

    def footer(self):
        message = "\n############################################\n"
        logging.info(message)


def download_img(img_url: str, img_name: str):
    response = requests.get(img_url)
    with open(img_name + ".jpg", "wb") as f:
        f.write(response.content)
    return img_name + ".jpg"


class url_img:
    def __init__(self, url: str, img_name: str):
        self.url = url
        self.img_name = img_name

    def download(self):
        downloaded_img_name = f"{self.img_name}.jpg"
        response = requests.get(self.url)
        with open(downloaded_img_name, "wb") as f:
            f.write(response.content)
        return downloaded_img_name
