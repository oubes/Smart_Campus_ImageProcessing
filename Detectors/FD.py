from src.face_detector import face_detector
import face_detection
import numpy as np


class fd_model(face_detector):
    """A subclass of face_detector that uses the face_recognition library."""

    def detector(self, gray_img: np.ndarray, detector_config: dict, img: np.ndarray):
        """Detect faces in the grayscale image using the face_recognition library and the detector configuration."""
        detector, confidence_threshold, nms_iou_threshold = detector_config
        detector_name = detector_config[0]
        detector = face_detection.build_detector(
            detector,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        detections = detector.detect(img)
        face_locations = []
        for face in detections:
            x1, y1, x2, y2, _ = face.astype(int)
            face_locations.append([x1, y1, x2, y2])
        # face_locations
        return face_locations, detector_name, detector_config
