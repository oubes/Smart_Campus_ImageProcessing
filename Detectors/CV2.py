from detection import face_detector
import cv2 as cv
import numpy as np
from vars import config

class cv2_model(face_detector):
    """A subclass of face_detector that uses the OpenCV library."""
    def detector(self, gray_img: np.ndarray, detector_config: dict, rgb_img: np.ndarray):
        """Detect faces in the grayscale image using the OpenCV library and the detector configuration."""
        sf = config["DetectorConfig"]["CV2"]["scaleFactor"]
        min_nh = config["DetectorConfig"]["CV2"]["minNeighbors"]
        min_win_size = tuple(detector_config["CV2"]["minSize"])
        detector_name = 'cv2'
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detections = face_cascade.detectMultiScale(gray_img, scaleFactor=sf, minNeighbors=min_nh, minSize=min_win_size)
        face_locations = []
        for face in detections:
            x, y, w, h = face
            face_locations.append([x, y, x+w, y+h])
        return face_locations, detector_name, detector_config
