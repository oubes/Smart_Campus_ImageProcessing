import logging
from typing import Tuple
import cv2 as cv
import os
from datetime import datetime
import numpy as np
import requests


class dir:
    def __init__(self, path: str, dir_name: str):
        self.dir_name = dir_name
        self.path = path
        self.dir_path = os.path.join(self.path, self.dir_name)

    def create(self):
        if os.path.exists(self.dir_path) is not True:
            os.makedirs(self.dir_path)


def read(img_input: str, gray=False) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(img_input, str):
        bgr_img = cv.imread(img_input)
        if bgr_img is None:
            raise FileNotFoundError(f"No image found at {img_input}")
    # If the input is a numpy array, assume it's an image
    elif isinstance(img_input, np.ndarray):
        bgr_img = img_input
    else:
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
