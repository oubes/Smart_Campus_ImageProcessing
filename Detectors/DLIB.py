from detection import face_detector
import face_recognition
import numpy as np

class fr_dlib_model(face_detector):
    """A subclass of face_detector that uses the face_recognition library."""
    def detector(self, gray_img: np.ndarray, detector_config: tuple, rgb_img: np.ndarray):
        """Detect faces in the grayscale image using the face_recognition library and the detector configuration."""
        upsampling, model_type = detector_config
        detector_name = 'fr_dlib'
        detections = face_recognition.face_locations(rgb_img, upsampling, model_type)
        face_locations = []
        for face in detections:
            y1, x2, y2, x1 = face
            face_locations.append([x1, y1, x2, y2])
        return face_locations, detector_name, detector_config