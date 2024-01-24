from detection import face_detector
from mtcnn import MTCNN
import numpy as np

class mtcnn_model(face_detector):
    def detector(self, gray_img: np.ndarray, detector_config: tuple, rgb_img: np.ndarray):
        min_face_size, steps_threshold, scale_factor = detector_config
        model = MTCNN(min_face_size=min_face_size, steps_threshold=steps_threshold, scale_factor=scale_factor)
        faces = model.detect_faces(rgb_img)
        detector_name = 'mtcnn'
        face_locations = []
        for facial_info in faces:
            facial_area = facial_info['box']
            x, y, w, h = facial_area
            face_locations.append([x, y, x+w, y+h])
        return face_locations, detector_name, detector_config