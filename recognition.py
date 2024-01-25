import os
import cv2 as cv
import vars
import toolbox
import time
from abc import ABC, abstractclassmethod
import pickle
import numpy as np

# def img_preprocessing():
#     pass
class face_recognizer(ABC):
    def __init__(self, detector_name, recognizer_config):
        self.file_directory = os.path.dirname(os.path.abspath(__file__))
        self.unlabeled_path = os.path.join(self.file_directory, vars.file_config.input_imgs_dir)
        self.labeled_path = os.path.join(self.file_directory, vars.file_config.labeled_dataset_dir)
        self.detector_name = detector_name
        self.threshold, self.config = recognizer_config
        self.config, self.encoding_update = self.config

    def run(self):
        t1 = time.perf_counter()
        self.initialize_face_locations_and_encodings()
        t2 = time.perf_counter()
        self.process_profiles()
        t3 = time.perf_counter()
        self.print_times(t1, t2, t3)
        self.output_img_handler()  

    @abstractclassmethod
    def encoder(self, img, face_locations, re_sample, model):
        pass

    def initialize_face_locations_and_encodings(self):
        self.fl, self.face_locations = self.detection(self.detector_name)
        self.unlabeled_face_encoded_img, self.unlabeled_face_img = self.img_encoding(self.unlabeled_path, vars.file_config.input_img_name, self.config, self.face_locations)
        self.best_match_confidences = [0] * len(self.unlabeled_face_encoded_img)
        self.best_match_names = ['Unknown'] * len(self.unlabeled_face_encoded_img)

    @abstractclassmethod
    def detection(self, detector_name):
        pass

    def img_encoding(self, path, input_img_name, config, face_locations=None, full_img=False):
        re_sample, model = config
        img = toolbox.img().read(os.path.join(path, input_img_name))
        if full_img:
            height, width, _ = img.shape
            face_locations = [(0, width, height, 0)]
        t1 = time.perf_counter()
        face_encoded_img = self.encoder(img, face_locations, re_sample, model) #dlib
        t2 = time.perf_counter()
        # print(t2 - t1)
        return face_encoded_img, img

    def process_profiles(self):
        for person in os.listdir(self.labeled_path):
            person_path = os.path.join(self.labeled_path, person, 'cropped_img')
            encoded_images = self.create_encoded_file(person_path, person)
            self.compare_and_update_best_match(encoded_images, person)

    def create_encoded_file(self, person_path, person):
        encoded_images = []
        if (not os.path.exists(f'{os.path.join(person_path, person)}_encoded.pkl')) or (self.encoding_update == True):
            encoded_images = self.read_encoded_file(person_path, person)
        else:
            print(f"File {person}_encoded.pkl already exists.")
            with open(f'{os.path.join(person_path, person)}_encoded.pkl', 'rb') as f:
                encoded_images = pickle.load(f)
        return encoded_images

    def read_encoded_file(self, person_path, person):
        encoded_images = []
        for labeled_face_name in [f for f in os.listdir(person_path) if os.path.splitext(f)[1] != '.pkl']:
            labeled_face_encoded_img, _ = self.img_encoding(person_path, labeled_face_name, self.config, full_img=True)
            encoded_images.append(labeled_face_encoded_img)
        with open(f'{os.path.join(person_path, person)}_encoded.pkl', 'wb') as f:
            pickle.dump(encoded_images, f)
        return encoded_images

    def compare_and_update_best_match(self, encoded_images, person):
        for encoding in encoded_images:
            for i in range(len(self.unlabeled_face_encoded_img)):
                matches, confidence = self.compare_faces(encoding, self.unlabeled_face_encoded_img[i], self.threshold)
                print(f'{self.fl[i]} -> Face({i+1}) <-> {person} <-> {matches} with confidence: {(confidence[0])*100: .2f}%')
                if matches[0] and (confidence[0]) > self.best_match_confidences[i]:
                    self.best_match_confidences[i] = confidence[0]
                    self.best_match_names[i] = person
    
    @abstractclassmethod
    def compare_faces(self, labeled_face_encoded_img, unlabeled_face_encoded_img, threshold):
        pass

    def print_times(self, t1, t2, t3):
        print(f'Detection Time: {(t2-t1):.3f} s')
        print(f'Recognition Time: {(t3-t2):.3f} s')     

    def output_img_handler(self):
        image = self.unlabeled_face_img.copy()
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
                fontScale = 1.4,
                color = (255, 0, 0), 
                thickness = 3,
                lineType = cv.LINE_AA
            )
            toolbox.img().draw_borders(image, [self.fl[i]])

        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)    
        resized_img = self.resize_image(img)
        toolbox.img().plot(resized_img, 'Image')

    def resize_image(self, image):
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

def Recognize(detector_name, recognizer_name):
    import vars
    from Recognizers import DLIB
    # OpenFace, FaceNet, DeepFace, ArcFace

    recognizers = {
        'DLIB': (DLIB.fr_dlib_model, vars.recognizer_config.fr_dlib),
        # 'OpenFace': (OpenFace.openface_model, vars.recognizer_config.openface),
        # 'FaceNet': (FaceNet.facenet_model, vars.recognizer_config.facenet),
        # 'DeepFace': (DeepFace.deepface_model, vars.recognizer_config.deepface),
        # 'ArcFace': (ArcFace.arcface_model, vars.recognizer_config.arcface)
    }

    if recognizer_name in recognizers:
        model_class, recognizer_config = recognizers[recognizer_name]
        model = model_class(
            detector_name = detector_name,
            recognizer_config = vars.recognizer_config.fr_dlib
        )
        model.run()
    else:
        raise ValueError(f"Unknown recognizer: {recognizer_name}")

# preprocessing (imp = 3/5: Quality)
# pre-encoding for labeled imgs [1 file -> person] (imp = 1/5: speed)
# code optimization and enhancing [main -> func(), config file:  -> Detector: , Generalization: Orgnization]
# Adding more models for recognition
# Connection with server
# Project discussion slides