from detection import face_detector
from retinaface import RetinaFace
import numpy as np
from vars import detector_config

class retinaface_model(face_detector):
    def detector(self, gray_img: np.ndarray, detector_config: tuple, rgb_img: np.ndarray):
        threshold = detector_config["RetinaFace"]["threshold"]
        upscaling = detector_config["RetinaFace"]["upsampleScale"] in [1]
        faces = RetinaFace.detect_faces(img_path=rgb_img, threshold=threshold, allow_upscaling = upscaling)
        detector_name = 'retinaface'
        face_locations = []
        for i, facial_info in faces.items():
            facial_area = facial_info['facial_area']
            x1, y1, x2, y2 = facial_area
            face_locations.append([x1, y1, x2, y2])
        return face_locations, detector_name, detector_config
