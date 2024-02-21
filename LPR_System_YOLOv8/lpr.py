from ultralytics import YOLO
import easyocr
import cv2
import os
import numpy as np
from vars import read_json
import matplotlib.pyplot as plt
import scipy.ndimage

class LPR:
    """A class for license plate recognition using YOLO and easyocr."""
    
    coco_model = YOLO('pretrained_models/yolov8s.pt')
    lpd_model = YOLO('pretrained_models/license_plate_detector.pt')
    readers = {}

    def __init__(self, img:str, lang: list, allow_list: list, allow_vehicles: list, config: dict):
        """Initialize the LPR class with the image, language, and allowed characters.

        Parameters:
        img (str): The path to the image file.
        lang (list): The list of languages to use for OCR.
        allow_list (list): The list of allowed characters for OCR.

        Attributes:
        img (str): The path to the image file.
        allow_list (list): The list of allowed characters for OCR.
        reader (easyocr.Reader): The OCR reader object.
        coco_model (YOLO): The YOLO model for car detection.
        lpd_model (YOLO): The YOLO model for license plate detection.
        allow_vehicles (list): The allowd types of vehicles for detection
        """
        self.img = img
        self.lang = lang
        print(lang)
        self.vehicles = allow_vehicles
        self.allow_list = allow_list
        print(allow_list)
        self.config = config
        if str(self.lang) not in LPR.readers:
            LPR.readers[str(self.lang)] = easyocr.Reader(self.lang, verbose=False, quantize=True)
        self.reader = LPR.readers[str(self.lang)]
    
    def read_img(self) -> np.ndarray:
        """Read the image file and convert it to RGB format.

        Returns:
        rgb_img (np.ndarray): The RGB image array.
        """
        bgr_img = cv2.imread(self.img)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def _crop_img(self, img: np.ndarray, box: list, style='xyxy') -> np.ndarray:
        """Crop the image according to the bounding box.

        Parameters:
        img (np.ndarray): The image array to crop.
        box (list): The coordinates of the bounding box.
        style (str): The style of the bounding box, either 'xyxy' or 'xywh'.

        Returns:
        crop_img (np.ndarray): The cropped image array.
        """
        try:
            if (style == 'xyxy') & (box is not None):
                x_min, y_min, x_max, y_max = map(int, box)
                crop_img = img[y_min:y_max, x_min:x_max]
                return crop_img
        
        except IndexError:
            print('Cropping Failed due to empty bounding box')
    
    def crop_imgs(self, imgs: list, boxes: list, style: str ='xyxy') -> list:
        """Crop multiple images from multiple images according to the bounding boxes.

        Parameters:
        imgs (list): The list of image arrays to crop.
        boxes (list): The list of bounding boxes.
        style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

        Returns:
        cropped_imgs (list): The list of cropped image arrays.
        """
        cropped_imgs = []
        for img, box in zip(imgs, boxes):
            cropped_imgs.append(self._crop_img(img=img, box=box, style=style))
        return cropped_imgs
    
    def detect_cars(self, img: np.ndarray) -> np.ndarray:
        """Detect cars in the image using the YOLO model.

        Parameters:
        img (np.ndarray): The image array to detect cars.

        Returns:
        car_boxes (np.ndarray): The list of bounding boxes for cars.
        """
        coco_results = self.coco_model(img)
        all_boxes = np.array(coco_results[0].boxes.xyxy)
        all_labels = np.array(coco_results[0].boxes.cls)
        desired_labels = np.array(self.vehicles) # 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus'
        idx = np.where(np.isin(all_labels, desired_labels))[0]
        return all_boxes[idx]
    
    def _detect_lp(self, img: np.ndarray) -> np.ndarray:
        """Detect the license plate in the image using the YOLO model.

        Parameters:
        img (np.ndarray): The image array to detect the license plate.

        Returns:
        lp_box (np.ndarray): The bounding box for the license plate.
        """
        try:
            lp_results = self.lpd_model(img)
            lp_box = lp_results[0].boxes.xyxy[0]
            lp_box = lp_box.detach().numpy().astype(np.int16)
            return lp_box
        except IndexError:
            print('No lp found')
    
    def detect_lps(self, imgs: list) -> list:
        """Detect the license plates in the images using the YOLO model.

        Parameters:
        imgs (list): The list of image arrays to detect the license plates.

        Returns:
        lps_box (np.ndarray): The list of bounding boxes for the license plates.
        """
        return [self._detect_lp(img=img) for img in imgs]
    
    def _recognize_lp(self, lp_img: list) -> np.ndarray:
        """Recognize the license plate number in the image using the easyocr reader.

        Parameters:
        lp_img (np.ndarray): The image array of the license plate.

        Returns:
        lp_text (np.ndarray): The license plate number.
        """
        try:
            if lp_img is not None:
                lp_img = np.array(lp_img)
                result = self.reader.readtext(lp_img, allowlist=self.allow_list)
                text = [res[1] for res in result]
                conf = np.array([res[2] for res in result])
                lp_text = "".join(text)
                print(f'LP Text: {lp_text}, confidence: {conf}')
                return lp_text
        except UnboundLocalError:
            pass
            
    def recognize_lps(self, lp_imgs: list) -> list:
        """Recognize the license plate numbers in the images using the easyocr reader.

        Parameters:
        lp_imgs (list): The list of image arrays of the license plates.

        Returns:
        lps (list): The list of license plate numbers.
        """
        return [self._recognize_lp(lp_img=lp_img) for lp_img in lp_imgs]
    
    def lp_alignment(self, lps_img: np.ndarray) -> np.ndarray:
        """
        Aligns the license plate images for better recognition by the easyocr reader.

        Args:
            lps_img (np.ndarray): An array of license plate images.

        Returns:
            np.ndarray: An array of aligned license plate numbers.
        """
        return lps_img
        
    def enhance(self, lps_imgs: list) -> list:
        """enhance the license plate images quality for more efficient recognition using the easyocr reader.

        Parameters:
        lp_imgs (list): The list of image arrays of the license plates.

        Returns:
        lps_imgs_enhanced (list): The list of license plate numbers.
        """
        upsample = self.config['LprConfig']['enhance']['upsample']
        lps_imgs_enhanced = []
        for img in lps_imgs:
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_upscale = scipy.ndimage.zoom(gray_img, upsample, order=3)
                img_alignment = self.lp_alignment(img_upscale)
                lps_imgs_enhanced.append(img_alignment)
            # plt.imshow(cv2.cvtColor(img_alignment, cv2.COLOR_BGR2RGB))
            # plt.show()
        return lps_imgs_enhanced
    
    def process_and_structure(self, license_plate_data):
        """Process and structure the output for license plate data.

        Parameters:
        license_plate_data (list): The list of arrays containing license plate text.

        Returns:
        processed_data (list): The list of processed license plate numbers.
        """
        return license_plate_data
    
    def compare(self, lps, lps_dB):
        
        """compare license plates text with the allowed ones in the database.

        Parameters:
        lps (list): The list of arrays of the license plates text.

        Returns:
        lps (list): The list of license plates authorized to enter.
        """
        return lps
    
    def run(self) -> list:
        """Run the LPR system on the image and return the license plate numbers.

        Returns:
        lps (list): The list of license plate numbers.
        """
        lps_dB = None
        rgb_img = self.read_img()
        car_boxes = self.detect_cars(rgb_img)
        cropped_cars = self.crop_imgs([rgb_img]*len(car_boxes), car_boxes)
        lps_box = self.detect_lps(cropped_cars)
        cropped_lps = self.crop_imgs(cropped_cars, lps_box)
        enhanced_lps = self.enhance(cropped_lps)
        lps = self.recognize_lps(enhanced_lps)
        lps_clean = self.process_and_structure(lps)
        lps_recognized = self.compare(lps_clean, lps_dB)
        return lps_recognized

# Start Testing Area
def dft(lang):
    """Run the LPR system on a list of images for a given language.

    Parameters:
    lang (str): The language to use for OCR.
    """
    config = read_json('config.json')
    langs = config['LprConfig']['langs']
    print(langs)
    allow_list = ""
    for lang in langs:
        allow_list += config['LprConfig']['allowLists'][lang]
    if 'ar' in langs or 'en' in langs:
        lps_list = []
        for idx, img in enumerate(os.listdir(f'imgs/{langs[0]}_lp')):
            print(f'Img number {idx+1}, Name: {img}')
            
            lpr_model = LPR(
                img=f'imgs/{langs[0]}_lp/{img}',
                lang=langs,
                allow_list=allow_list,
                allow_vehicles=config['LprConfig']['allowVehicles'],
                config=config
            )
            
            lps_list.append(lpr_model.run())
        flattened_list = [item for sublist in lps_list for item in sublist]
        print(flattened_list)
    else:
        print("unsupported language")
# End Testing Area

if __name__ == "__main__":
    
    dft('en')


