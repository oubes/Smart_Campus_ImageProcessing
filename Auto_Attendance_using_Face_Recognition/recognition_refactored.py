import time
import numpy as np
from typing import List, Tuple
import toolbox
from tasks import Detect
import vars
from abc import ABC, abstractmethod


class face_recognizer(ABC):
    def __init__(self, encoded_dict: dict[str, dict]):
        self.detector_name = vars.detector
        self.recognizer_name = vars.recognizer
        self.recognizer_config = vars.recognizer_config
        self.encoded_dict = encoded_dict
        self.d_encoding_time = 0

    def Recognize(self, unlabeled_img_url):
        # Initializing numpy arrays and encodings
        t1 = time.perf_counter()
        unlabeled_fl, _, unlabeled_rgb_img = Detect(self.detector_name, unlabeled_img_url)
        unlabeled_fl = toolbox.points2rotation_format(unlabeled_fl)

        t1_1 = time.perf_counter()
        unlabeled_encoded_faces = self.encoder(unlabeled_rgb_img, unlabeled_fl, self.recognizer_config)
        t1_2 = time.perf_counter()
        self.d_encoding_time = t1_2 - t1_1

        self.best_match_confidences = np.zeros(len(unlabeled_encoded_faces))
        self.best_match_names = np.full(len(unlabeled_encoded_faces), 'Unknown', dtype='U20')

        t2 = time.perf_counter()

        IDs = np.array(list(self.encoded_dict.keys()))
        for ID in IDs:
            for labeled_encoded_face in self.encoded_dict[ID]:
                for i, unlabeled_encoded_face in enumerate(unlabeled_encoded_faces):
                    matches, confidence = self.compare_faces(labeled_encoded_face, unlabeled_encoded_face, self.recognizer_config["threshold"])
                    if matches[0] and (confidence[0]) > self.best_match_confidences[i]:
                        self.best_match_confidences[i] = confidence[0]
                        self.best_match_names[i] = ID
                    

        return IDs

    def add_labeled_encoded_entry(self, encoded_dict: dict, labeled_face_url: str, ID: str):
        # Add check for config
        if labeled_face_url in encoded_dict[ID].keys() and not(self.recognizer_config.encodingUpdate):
            print(f'No new images for {ID} were found')
        else:
            labeled_fl, _, labeled_rgb_img = Detect(self.detector_name, labeled_face_url)
            labeled_fl = toolbox.points2rotation_format(labeled_fl)
            labeled_encoded_face = self.encoder(labeled_rgb_img, labeled_fl, self.recognizer_config)
            if isinstance(labeled_encoded_face, np.ndarray):
                labeled_encoded_face = labeled_encoded_face.tolist()
            encoded_dict[ID][labeled_face_url] = labeled_encoded_face
            return encoded_dict


    @abstractmethod
    def encoder(self, image, face_locations, config) -> List[np.ndarray]:
        pass

    @abstractmethod
    def compare_faces(self, labeled_face_encoded_img, unlabeled_face_encoded_imgs, threshold) -> Tuple(List, np.ndarray):
        pass

    def update_dict(self, encoded_labeled_faces, ID: str):
        self.encoded_dict[ID] = encoded_labeled_faces
        print(self.encoded_dict)
