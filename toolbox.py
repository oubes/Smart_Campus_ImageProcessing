import logging
import cv2 as cv
import os
from datetime import datetime
import urllib.request
import math
import numpy as np

class dir:
    def __init__(self, path: str, dir_name: str):
        self.dir_name = dir_name
        self.path = path
        self.dir_path = os.path.join(self.path, self.dir_name)
    def create(self):
        if os.path.exists(self.dir_path) is not True:
            os.makedirs(self.dir_path)

class img:
    def __init__(self):
        pass
    
    def read(self, img_path: str, gray=False):
        bgr_img = cv.imread(img_path)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        if gray is True:
            try:
                gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
                return rgb_img, gray_img
            except:
                print('No img was found in the given path')
        else:
            return rgb_img
    
    def create(self, new_img_path: str, img: np.ndarray):
        cv.imwrite(f'{new_img_path}.jpg', img)

    def remove(self, img_path: str):
        try:
            os.remove(img_path)
        except:
            print("An error occurred!")

    def crop_imgs(self, img: np.ndarray, new_img_path: str, imgs_coord: list):
        bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        for i, (x1, y1, x2, y2) in enumerate(imgs_coord):
            clip = bgr_img[y1:y2, x1:x2]
            self.create(f'{new_img_path}_{i+1}', clip)
    
    def draw_borders(self, img: np.ndarray, imgs_coord: list):
        for i, (x1, y1, x2, y2) in enumerate(imgs_coord):
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def plot(self, img: np.ndarray, title: str):
        try:
            cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imshow(f'{title}', img)
            cv.waitKey(0); cv.destroyAllWindows()
        except:
            print('No img was found in the given path')

    def points2rotation_format(self, fl):
        face_locations = []
        for face_location in fl:
            facial_area = face_location
            x1, y1, x2, y2 = facial_area
            face_locations.append([y1, x2, y2, x1])  
        return face_locations

class logger:
    def __init__(self):
        logging.basicConfig(filename='log.log', level=logging.INFO)
        self.current_time = f'{datetime.now().replace(microsecond=0)}'
    def add(self, message: str):
        logging.info(f'{self.current_time}: {message}')
    def footer(self):
        message = '\n############################################\n'
        logging.info(message)

class url_img:
    def __init__(self, url: str, img_name: str):
        self.url = url
        self.img_name = img_name
    def download(self):
        downloaded_img_name = f'{self.img_name}.jpg'
        urllib.request.urlretrieve(self.url, downloaded_img_name)
        return downloaded_img_name

