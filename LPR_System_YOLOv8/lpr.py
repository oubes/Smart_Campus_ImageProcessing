from ultralytics import YOLO
import easyocr
import cv2
import os
import numpy as np
from vars import read_json
# import matplotlib.pyplot as plt

class LPR:
    """A class for license plate recognition using YOLO and easyocr."""
    
    coco_model = YOLO('pretrained_models/yolov8s.pt')
    lpd_model = YOLO('pretrained_models/license_plate_detector.pt')
    _reader = None
    _lang = None
    
    @property
    def reader(self):
        if self._reader is None or self._lang != self.lang:
            self._reader = easyocr.Reader(self.lang, verbose=False, quantize=True)
            self._lang = self.lang
        return self._reader

    def __init__(self, img, lang: list, allow_list: list):
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
        """
        self.img = img
        self.lang = lang
        self.allow_list = allow_list
    
    def read_img(self):
        """Read the image file and convert it to RGB format.

        Returns:
        rgb_img (np.ndarray): The RGB image array.
        """
        bgr_img = cv2.imread(self.img)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def _crop_img(self, img, box, style='xyxy'):
        """Crop the image according to the bounding box.

        Parameters:
        img (np.ndarray): The image array to crop.
        box (list): The coordinates of the bounding box.
        style (str): The style of the bounding box, either 'xyxy' or 'xywh'.

        Returns:
        rescaled_img (np.ndarray): The rescaled cropped image array.
        """
        try:
            if (style == 'xyxy') & (box is not None):
                x_min, y_min, x_max, y_max = map(int, box)
                crop_img = img[y_min:y_max, x_min:x_max]
                height, width, _ = crop_img.shape
                scale = 2
                new_height = height * scale
                new_width = width * scale
                matrix = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
                rescaled_img = cv2.warpAffine(crop_img, matrix, (new_width, new_height))
                return rescaled_img
        
        except IndexError:
            print('Cropping Failed due to empty bounding box')
    
    def crop_imgs(self, img: np.ndarray, boxes: list, style: str ='xyxy'):
        """Crop multiple images according to the bounding boxes.

        Parameters:
        img (np.ndarray): The image array to crop.
        boxes (list): The list of bounding boxes.
        style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

        Returns:
        cropped_imgs (list): The list of cropped image arrays.
        """
        cropped_imgs = []
        for box in boxes:
            cropped_imgs.append(self._crop_img(img=img, box=box, style=style))
        return cropped_imgs
    
    def crop_multi_imgs(self, imgs: list, boxes: list, style: str ='xyxy'):
        """Crop multiple images from multiple images according to the bounding boxes.

        Parameters:
        imgs (list): The list of image arrays to crop.
        boxes (list): The list of bounding boxes.
        style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

        Returns:
        lps (list): The list of cropped image arrays.
        """
        lps = []
        for img, box in zip(imgs, boxes):
            lps.append(self._crop_img(img=img, box=box, style=style))
        return lps
    
    def detect_cars(self, img: np.ndarray):
        """Detect cars in the image using the YOLO model.

        Parameters:
        img (np.ndarray): The image array to detect cars.

        Returns:
        car_boxes (list): The list of bounding boxes for cars.
        """
        coco_results = self.coco_model(img)
        all_boxes = coco_results[0].boxes.xyxy
        all_labels = coco_results[0].boxes.cls
        desired_labels = (1, 2, 3, 5) # 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus'
        car_boxes = [box for box, label in zip(all_boxes, all_labels) if label in desired_labels]
        return car_boxes
    
    def _detect_lp(self, img: np.ndarray):
        """Detect the license plate in the image using the YOLO model.

        Parameters:
        img (np.ndarray): The image array to detect the license plate.

        Returns:
        lp_box (list): The bounding box for the license plate.
        """
        try:
            lp_results = self.lpd_model(img)
            lp_box = lp_results[0].boxes.xyxy[0]
            return lp_box
        except IndexError:
            print('No lp found')
    
    def detect_lps(self, imgs: list):
        """Detect the license plates in the images using the YOLO model.

        Parameters:
        imgs (list): The list of image arrays to detect the license plates.

        Returns:
        lps_box (list): The list of bounding boxes for the license plates.
        """
        lps_box = []
        for img in imgs:
            lps_box.append(self._detect_lp(img=img))
        return lps_box
    
    def _recognize_lp(self, lp_img: list):
        """Recognize the license plate number in the image using the easyocr reader.

        Parameters:
        lp_img (np.ndarray): The image array of the license plate.

        Returns:
        lp_text (str): The license plate number.
        """
        if lp_img is not None:
            lp_text = ""
            result = self.reader.readtext(lp_img, allowlist=self.allow_list)
            for res in result:
                _, text, conf = res
                lp_text += text
            print(f'LP Text: {lp_text}, confidence: {conf}')
            return lp_text
            
    def recognize_lps(self, lp_imgs: list):
        """Recognize the license plate numbers in the images using the easyocr reader.

        Parameters:
        lp_imgs (list): The list of image arrays of the license plates.

        Returns:
        lps (list): The list of license plate numbers.
        """
        lps = []
        for lp_img in lp_imgs:
            lps.append(self._recognize_lp(lp_img=lp_img))
        return lps
    
    def run(self):
        """Run the LPR system on the image and return the license plate numbers.

        Returns:
        lps (list): The list of license plate numbers.
        """
        rgb_img = self.read_img()
        car_boxes = self.detect_cars(rgb_img)
        cropped_cars = self.crop_imgs(rgb_img, car_boxes)
        lps_box = self.detect_lps(cropped_cars)
        cropped_lps = self.crop_multi_imgs(cropped_cars, lps_box)
        return self.recognize_lps(cropped_lps)

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
        allow_list += config['LprConfig']['allowlists'][lang]
    print(allow_list)
    if lang == 'ar' or 'en':
        lps_list = []
        for idx, img in enumerate(os.listdir(f'imgs/{lang}_lp')):
            print(f'Img number {idx+1}, Name: {img}')
            lpr_model = LPR(img=f'imgs/{lang}_lp/{img}', lang=[lang], allow_list=allow_list)
            lps_list.append(lpr_model.run())
        flattened_list = [item for sublist in lps_list for item in sublist]
        print(flattened_list)
    else:
        print("unsupported language")

if __name__ == "__main__":
    
    dft('en')


