import sys
import cv2
# import json
from pathlib import Path


# import cv2
# from easyocr.utils import group_text_box, word_segmentation
# import tensorflow as tf
# import keras

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.lp_detection as lp_detection
import src.lp_recognition as lp_recognition
import src.lp_preprocessing as lp_preprocessing
import src.lp_data_processing as lp_data_processing
import time
from src.vars import read_json
from src.lp_predict import lp_predict
from src.lp_new import imageToSymbol, preprocess


class LPR:
    """A class for license plate recognition using YOLO and easyocr."""

    def __init__(self, img: str, config: dict):
        """Initialize the LPR class with the image, language, allowVehicles, and allowed characters.

        Parameters:
        img (str): The path to the image file.
        config (dict): Contains all the required config for license plate detection

        """
        self.img = img
        self.lang = config["LprConfig"]["lang"]
        self.vehicles = config["LprConfig"]["allowVehicles"]
        self.allow_list = config["LprConfig"]["allowLists"][self.lang]
        self.enhance = config["LprConfig"]["enhance"]

    def run(self, test_mode: bool = False) -> list:
        """Run the LPR system on the image and return the license plate numbers.

        Returns:
        lps (list): The list of license plate numbers.
        """

        rgb_img = lp_preprocessing.read_img(self.img)
        if rgb_img is None:
            raise FileNotFoundError
        car_boxes = lp_detection.detect_cars(rgb_img, self.vehicles)
        if car_boxes is None:
            raise RuntimeError("No cars detected in the image.")
        cropped_cars = lp_preprocessing.crop_imgs(
            [rgb_img] * len(car_boxes), car_boxes, type="car"
        )
        if cropped_cars is None:
            raise ValueError("Cropping cars failed.")
        lps_box = lp_detection.detect_lps(cropped_cars)
        if lps_box is None:
            raise ValueError("Could not detect license plates.")

        cropped_lps = lp_preprocessing.crop_imgs(cropped_cars, lps_box, type="lp")
        if cropped_lps is None:
            raise ValueError("Cropping license plates failed.")

        # enhanced_lps = []
        # print(f'cropped_lps.shape = {len(cropped_lps)}')
        # lps = []
        # for cropped_lp in cropped_lps:
        #     if cropped_lp is None:
        #         continue
        #     # lps.append(lp_predict(cropped_lp))
        #     OpenedThresh, _ = preprocess(cropped_lp)
        #
        #     lps = lp_recognition._recognize_lp(OpenedThresh, allow_list=self.allow_list)
        #
        # return lps


            # enhanced_lps.append(preprocess(cropped_lp)[0])
            # print(f'lps in run_lpr = {lps}')
            # print('Line 68')
        # return lps



        # enhanced_lps = lp_preprocessing.preprocessing(
        #     cropped_lps, self.enhance, test_mode
        # )
        # if enhanced_lps is None:
        #     raise RuntimeError("License plates preprocessing failed.")


        # lps = lp_recognition.recognize_lps(enhanced_lps, self.allow_list)
        #
        # if lps is None:
        #     raise ValueError("Could not recognize license plates.")

        for cropped_lp in cropped_lps:
            lps = lp_predict(cropped_lp)
        # lps_clean = lp_data_processing.process_and_structure(lps)
        # if lps_clean is None:
        #     raise NameError("Could not process license plates data.")
        # return lps_clean[0]["lp"]
        return lps

# Start Testing Area
def dft(lang):
    """Run the LPR system on a list of images for a given language.

    Parameters:
    lang (str): The language to use for OCR.
    """
    import os
    import numpy as np

    if np.isin(lang, ["en", "ar"]):
        for idx, img in enumerate(os.listdir(f"imgs/{lang}_lp")):
            print(f'Detecting license plate on {img}...')
            t1 = time.time()
            lpr_model = LPR(img=f"imgs/{lang}_lp/{img}", config=config)
            lp = lpr_model.run(test_mode=config["LprConfig"]["testMode"])
            t2 = time.time()
            print(
                f"Img number {idx+1}, Name: {img}, lps: {lp}, time taken: {(t2-t1):.1f} s"
            )

    else:
        print("unsupported language")


# End Testing Area

if __name__ == "__main__":
    config = read_json("config/lp_config.json")
    lang = config["LprConfig"]["lang"]
    dft(lang)

    # import tensorflow as tf
    # from tensorflow import keras
    #
    # number_model = tf.keras.models.load_model('pretrained_models/number_model.keras')
    # letter_model = tf.keras.models.load_model('pretrained_models/letter_model.keras')

    

