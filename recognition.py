import os
import cv2 as cv
import vars
import toolbox
import time
from abc import ABC, abstractclassmethod
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

# def img_preprocessing():
#     pass
class face_recognizer(ABC):
    def __init__(self, detector_name, recognizer_config):
        self.file_directory = os.path.dirname(os.path.abspath(__file__))
        # self.unlabeled_path = os.path.join(self.file_directory, vars.file_config.input_imgs_dir)
        self.labeled_path = os.path.join(self.file_directory, vars.file_config.labeled_dataset_dir)
        self.detector_name = detector_name
        self.threshold, self.config = recognizer_config
        self.config, self.encoding_update = self.config

    def run(self):
        t1 = time.perf_counter()
        self._initialize_face_locations_and_encodings()
        t2 = time.perf_counter()
        self._process_profiles()
        t3 = time.perf_counter()
        self._print_times(t1, t2, t3)
        known_names = self._output_img_handler()
        return known_names

    def _initialize_face_locations_and_encodings(self):
        self.fl, self.face_locations, self.rgb_img = self.detection(self.detector_name)
        self.unlabeled_face_encoded_img, self.unlabeled_face_img = self._img_encoding(self.rgb_img, self.config, self.face_locations)
        self.best_match_confidences = [0] * len(self.unlabeled_face_encoded_img)
        self.best_match_names = ['Unknown'] * len(self.unlabeled_face_encoded_img)

    @abstractclassmethod
    def detection(self, detector_name):
        pass

    def _img_encoding(self, img, config, face_locations=None, full_img=False):
        self.re_sample, self.model = config
        if full_img:
            height, width, _ = img.shape
            face_locations = [(0, width, height, 0)]
        face_encoded_img = self.encoder(img, face_locations, self.re_sample, self.model) #dlib
        return face_encoded_img, img
    
    @abstractclassmethod
    def encoder(self, img, face_locations, re_sample, model):
        pass

    def _process_profiles(self):
        for person in os.listdir(self.labeled_path):
            person_path = os.path.join(self.labeled_path, person, 'cropped_img')
            encoded_images_dict = self._create_encoded_file(person_path, person)
            # Extract only the encoded images from the dictionary
            encoded_images = list(encoded_images_dict.values())
            self._compare_and_update_best_match(encoded_images, person)

    def _create_encoded_file(self, person_path, person):
        data = {}
        encoded_file_path = f'{os.path.join(person_path, person)}_encoded.json'
        if os.path.exists(encoded_file_path):
            with open(encoded_file_path, 'r') as f:
                data = json.load(f)
            current_images = set([f for f in os.listdir(person_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']])
            if current_images == set(data.get('encoded_images', {}).keys()) and not self.encoding_update and data.get('config') == {'re_sample': self.re_sample, 'model': self.model}:
                print(f"File {person}_encoded.json already exists and no new images or config changes found.")
            else:
                print(f"New or updated images found or config changed for {person}. Updating encodings.")
                data['encoded_images'] = self._read_and_encode_images(person_path, person)
                data['config'] = {'re_sample': self.re_sample, 'model': self.model}
                data['person_name'] = person
                with open(encoded_file_path, 'w') as f:
                    json.dump(data, f, cls=NumpyEncoder)
        else:
            print(f"No encoded file found for {person}. Creating new encodings.")
            data['encoded_images'] = self._read_and_encode_images(person_path, person)
            data['config'] = {'re_sample': self.re_sample, 'model': self.model}
            data['person_name'] = person
            with open(encoded_file_path, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder)
        if person == 'Eslam Hakel':
            data['encoded_images']
        return data['encoded_images']

    def _read_and_encode_images(self, person_path, person):
        encoded_images = {}
        for labeled_face_name in [f for f in os.listdir(person_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]:
            labeled_face_encoded_img, _ = self._img_encoding(person_path, labeled_face_name, self.config, full_img=True)
            # Check if labeled_face_encoded_img is a numpy array before converting
            if isinstance(labeled_face_encoded_img, np.ndarray):
                labeled_face_encoded_img = labeled_face_encoded_img.tolist()
            encoded_images[labeled_face_name] = labeled_face_encoded_img
        return encoded_images


    def _compare_and_update_best_match(self, encoded_images, person):
        for encoding in encoded_images:
            for i in range(len(self.unlabeled_face_encoded_img)):
                matches, confidence = self.compare_faces(encoding, self.unlabeled_face_encoded_img[i], self.threshold)
                # print(f'{self.fl[i]} -> Face({i+1}) <-> {person} <-> {matches} with confidence: {(confidence[0])*100: .2f}%')
                if matches[0] and (confidence[0]) > self.best_match_confidences[i]:
                    self.best_match_confidences[i] = confidence[0]
                    self.best_match_names[i] = person
    
    @abstractclassmethod
    def compare_faces(self, labeled_face_encoded_img, unlabeled_face_encoded_img, threshold):
        pass

    def _print_times(self, t1, t2, t3):
        print(f'Detection Time: {(t2-t1):.3f} s')
        print(f'Recognition Time: {(t3-t2):.3f} s')     

    def _output_img_handler(self):
        image = self.rgb_img
        for i in range(len(self.unlabeled_face_encoded_img)):
            if self.best_match_names[i] != 'Unknown':
                print(f'The correct identity for Face({i+1}) is {self.best_match_names[i]} with confidance: {(self.best_match_confidences[i])*100: .2f}%')  # Print the person's name
            else:
                print(f'The identity for Face({i+1}) is Unknown')
            image = cv.putText(
                img = image,
                text = self.best_match_names[i]+f",{self.best_match_confidences[i]*100: .2f}%",  # Use the person's name
                org = (self.fl[i][0]-80, self.fl[i][1]-30),
                fontFace = cv.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = (255, 0, 0), 
                thickness = 1,
                lineType = cv.LINE_AA
            )
            toolbox.img().draw_borders(image, [self.fl[i]])

        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)    
        resized_img = self._resize_image(img)
        # toolbox.img().plot(resized_img, 'Image')
        return [name for name in self.best_match_names if name != 'Unknown']

    def _resize_image(self, image):
        h, w, _ = image.shape
        ratio = ratio1 = ratio2 = 1
        while (h > 1020) or (w > 1920):
            if(h > 1020):
                ratio1 = (1000 / h)*0.8
            if(w > 1920):
                ratio2 = (1920 / w)*0.8
            if(h > 1020 or w > 1920):
                ratio = min(ratio1, ratio2)
                h = h * ratio; w = w * ratio
        return cv.resize(image, (int(w), int(h)), interpolation = cv.INTER_AREA)



# preprocessing (imp = 3/5: Quality)
# code optimization and enhancing [config file -> Detector]
# Adding more models for recognition
# Connection with server
# Project discussion slides