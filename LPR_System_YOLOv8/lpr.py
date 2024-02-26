import detection, recognition, lp_preprocessing, lp_data_processing

from vars import read_json

class LPR:
    """A class for license plate recognition using YOLO and easyocr."""

    def __init__(self, img: str, config: dict):
        """Initialize the LPR class with the image, language, allowVehicles, and allowed characters.

        Parameters:
        img (str): The path to the image file.
        config (dict): Contains all the required config for license plate detection

        """
        self.img = img
        self.lang = config['LprConfig']['lang']
        self.vehicles = config['LprConfig']['allowVehicles']
        self.allow_list = config['LprConfig']['allowLists'][self.lang]
        self.enhance = config['LprConfig']['enhance']

    def run(self, test_mode: bool = False) -> list:
        """Run the LPR system on the image and return the license plate numbers.

        Returns:
        lps (list): The list of license plate numbers.
        """
        rgb_img = lp_preprocessing.read_img(self.img)
        car_boxes = detection.detect_cars(rgb_img, self.vehicles)
        cropped_cars = lp_preprocessing.crop_imgs([rgb_img]*len(car_boxes), car_boxes)
        lps_box = detection.detect_lps(cropped_cars)
        cropped_lps = lp_preprocessing.crop_imgs(cropped_cars, lps_box)
        enhanced_lps = lp_preprocessing.preprocessing(cropped_lps, self.enhance, test_mode)
        lps = recognition.recognize_lps(enhanced_lps, self.allow_list)
        lps_clean = lp_data_processing.process_and_structure(lps)
        return lps_clean
        
# Start Testing Area
def dft(lang):
    """Run the LPR system on a list of images for a given language.

    Parameters:
    lang (str): The language to use for OCR.
    """
    import os
    import numpy as np
    
    if np.isin(lang, ['en', 'ar']):
        lps_list = []
        for idx, img in enumerate(os.listdir(f'imgs/{lang}_lp')):
            print(f'Img number {idx+1}, Name: {img}')
            
            lpr_model = LPR(
                img=f'imgs/{lang}_lp/{img}',
                config=config
            )
            
            lps_list.append(lpr_model.run(test_mode=config['LprConfig']['testMode']))
        flattened_list = [item for sublist in lps_list for item in sublist]
        print(flattened_list)
    else:
        print("unsupported language")
# End Testing Area

if __name__ == "__main__":
    config = read_json('config.json')
    lang = config['LprConfig']['lang']
    dft(lang)


