from ultralytics import YOLO
import easyocr
import cv2
import os
# import numpy as np
# import matplotlib.pyplot as plt


class LPR:
    def __init__(self, img, lang: list, allow_list: list):
        self.img = img
        self.allow_list = allow_list
        self.reader = easyocr.Reader(lang, verbose=False, quantize=True)
        self.coco_model = YOLO(r'pretrained_models/yolov8s.pt')
        self.lpd_model = YOLO(r'pretrained_models/license_plate_detector.pt')
    
    def read_img(self):
        bgr_img = cv2.imread(self.img)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    def _crop_img(self, img, box, style='xyxy'):
        try:
            if (style == 'xyxy') & (box is not None):
                x1, y1, x2, y2 = map(int, box)
                crop_img = img[y1:y2, x1:x2]
                rescaled_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                return rescaled_img
        
        except IndexError:
            print('Cropping Failed due to empty bounding box')
    
    def crop_imgs(self, img, boxs, style='xyxy'):
        cropped_imgs = []
        for box in boxs:
            cropped_imgs.append(self._crop_img(img=img, box=box, style=style))
        return cropped_imgs
    
    def crop_multi_imgs(self, imgs, boxs, style='xyxy'):
        lps = []
        for img, box in zip(imgs, boxs):
            lps.append(self._crop_img(img=img, box=box, style=style))
        return lps
    
    def detect_cars(self, img):
        coco_results = self.coco_model(img)
        all_boxes = coco_results[0].boxes.xyxy
        all_labels = coco_results[0].boxes.cls
        car_boxes = [box for box, label in zip(all_boxes, all_labels) if label == 2]
        return car_boxes
    
    def _detect_lp(self, img):
        try:
            lp_results = self.lpd_model(img)
            lp_box = lp_results[0].boxes.xyxy[0]
            return lp_box
        except IndexError:
            print('No lp found')
    
    def detect_lps(self, imgs):
        lps_box = []
        for img in imgs:
            lps_box.append(self._detect_lp(img=img))
        return lps_box
    
    def _recognize_lp(self, lp_img: list):
        if lp_img is not None:
            result = self.reader.readtext(lp_img, allowlist=self.allow_list)
            for idx, res in enumerate(result):
                _, text, conf = res
                print(f'{idx+1}: Text: {text}, Conf: {conf}')
            
    def recognize_lps(self, lp_imgs: list):
        lps = []
        for lp_img in lp_imgs:
            lps.append(self._recognize_lp(lp_img=lp_img))
        return lps
    
    def run(self):
        rgb_img = self.read_img()
        car_boxes = self.detect_cars(rgb_img)
        cropped_cars = self.crop_imgs(rgb_img, car_boxes)
        lps_box = self.detect_lps(cropped_cars)
        cropped_lps = self.crop_multi_imgs(cropped_cars, lps_box)
        self.recognize_lps(cropped_lps)
        
allowlist_en = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ -'
allowlist_ar = 'أبجدةهوزحطيكلمنسعفصقرشتثخذضظغ٠١٢٣٤٥٦٧٨٩ -'

def dft(lang):
    if lang == 'ar':
        for idx, img in enumerate(os.listdir('imgs/ar_lp')):
            print(f'Img number {idx+1}, Name: {img}')
            lpr_model = LPR(img=f'imgs/ar_lp/{img}', lang=['ar'], allow_list=allowlist_ar)
            lpr_model.run()

    elif lang == 'en':
        for idx, img in enumerate(os.listdir('imgs/en_lp')):
            print(f'Img number {idx+1}, Name: {img}')
            lpr_model = LPR(img=f'imgs/en_lp/{img}', lang=['en'], allow_list=allowlist_en)
            lpr_model.run()

dft('en')