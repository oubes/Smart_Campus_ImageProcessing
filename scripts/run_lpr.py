import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.lp_detection as lp_detection
import src.lp_recognition as lp_recognition
import src.lp_preprocessing as lp_preprocessing
import src.lp_data_processing as lp_data_processing
import time
from src.vars import read_json


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
        enhanced_lps = lp_preprocessing.preprocessing(
            cropped_lps, self.enhance, test_mode
        )
        if enhanced_lps is None:
            raise RuntimeError("License plates preprocessing failed.")

        lps = lp_recognition.recognize_lps(enhanced_lps, self.allow_list)
        if lps is None:
            raise ValueError("Could not recognize license plates.")
        lps_clean = lp_data_processing.process_and_structure(lps)
        if lps_clean is None:
            raise NameError("Could not process license plates data.")
        return lps_clean[0]["lp"]

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
