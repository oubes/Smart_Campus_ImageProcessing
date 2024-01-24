from detection import face_detector
from ultralytics import YOLO
import numpy as np

class yolo8_model(face_detector):
    def detector(self, gray_img: np.ndarray, detector_config: tuple, rgb_img: np.ndarray):
        threshold = detector_config
        model = YOLO("yolov8n-face.pt")
        results = model(rgb_img)
        conf = results[0].boxes.conf
        fl = np.array(results[0].boxes.xyxy.cpu().int())
        face_locations = [box for i, box in enumerate(fl) if conf[i] > threshold]
        detector_name = 'yolo8'
        return face_locations, detector_name, detector_config
