import time
import json
import numpy as np
import utils.toolbox as toolbox
from typing import List, Tuple
from src.tasks import Detect
import src.vars as vars
from abc import ABC, abstractmethod


class face_recognizer(ABC):
    def __init__(self, recognizer_name, recognizer_config):
        self.detector_name = vars.detector
        self.recognizer_name = recognizer_name
        self.recognizer_config = recognizer_config
        self.d_encoding_time = 0

    def Recognize(self, unlabeled_img_url, encoded_dict):
        # Initializing numpy arrays and encodings
        t1 = time.perf_counter()
        unlabeled_fl, fc, unlabeled_rgb_img = Detect(
            self.detector_name, unlabeled_img_url
        )
        unlabeled_fl = toolbox.points2rotation_format(unlabeled_fl)

        t1_1 = time.perf_counter()
        unlabeled_encoded_faces = self.encoder(
            unlabeled_rgb_img, unlabeled_fl, self.recognizer_config
        )
        t1_2 = time.perf_counter()
        self.d_encoding_time = t1_2 - t1_1

        self.best_match_confidences = np.zeros(len(unlabeled_encoded_faces))
        self.best_match_names = np.full(
            len(unlabeled_encoded_faces), "Unknown", dtype="U36"
        )

        t2 = time.perf_counter()

        for encoded_dict_instance in encoded_dict:
            if (
                encoded_dict_instance["imgs"] is None
                or encoded_dict_instance["imgs"] == ""
            ):
                continue

            labeled_encoded_faces = json.loads(encoded_dict_instance["imgs"])
            for labeled_encoded_face_i in labeled_encoded_faces:
                labeled_encoded_face = np.array([labeled_encoded_face_i])
                for j, unlabeled_encoded_face_j in enumerate(unlabeled_encoded_faces):
                    unlabeled_encoded_face = np.array([unlabeled_encoded_face_j])
                    matches, confidence = self.compare_faces(
                        labeled_encoded_face,
                        unlabeled_encoded_face,
                        self.recognizer_config["threshold"],
                    )
                    if matches[0] and (confidence[0]) > self.best_match_confidences[j]:
                        self.best_match_confidences[j] = confidence[0]
                        self.best_match_names[j] = encoded_dict_instance["id"]

        t3 = time.perf_counter()
        self.print_times(t1, t2, t3, self.d_encoding_time)

        return [name for name in self.best_match_names if name != "Unknown"], fc

    def add_labeled_encoded_entry(self, labeled_face_url: str, encoded_dict: list):
        # Add check for config
        labeled_fl, face_count, labeled_rgb_img = Detect(
            self.detector_name, labeled_face_url
        )

        if face_count == 0:
            raise ValueError(f"No face found in image '{labeled_face_url}'")
            # if face_count > 1:
            # raise ValueError(f"Multiple faces found in image '{labeled_face_url}'")

        if len(encoded_dict) == 0:
            encoded_dict = []
        else:
            encoded_dict = list(json.loads(str(encoded_dict)))

        if self.detector_name == "RetinaFace":
            labeled_fl = toolbox.points2rotation_format(labeled_fl)
        labeled_encoded_face = self.encoder(
            labeled_rgb_img, labeled_fl, self.recognizer_config
        )

        if isinstance(labeled_encoded_face[0], np.ndarray):
            labeled_encoded_face = labeled_encoded_face[0].tolist()
        encoded_dict.append(labeled_encoded_face)
        return encoded_dict

    @abstractmethod
    def encoder(self, image, face_locations, config) -> List[np.ndarray]:
        pass

    @abstractmethod
    def compare_faces(
        self, labeled_face_encoded_img, unlabeled_face_encoded_imgs, threshold
    ) -> Tuple[List[bool], np.ndarray]:
        pass

    def print_times(self, t1, t2, t3, d_encoding_time):
        print(f"Detection Time: {(t2-t1-d_encoding_time):.3f} s")
        print(f"Dectection -> Encoding Time: {(d_encoding_time):.3f} s")
        print(f"Recognition Time: {(t3-t2):.3f} s")
