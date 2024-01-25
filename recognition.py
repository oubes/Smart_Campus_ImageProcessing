import os
import cv2 as cv
import vars
import toolbox
import time
from abc import ABC, abstractclassmethod
import pickle

# def img_preprocessing():
#     pass
class face_recognizer(ABC):
    def __init__(self, detector_name, recognizer_config):
        self.file_directory = os.path.dirname(os.path.abspath(__file__))
        self.unlabeled_path = os.path.join(self.file_directory, vars.file_config.input_imgs_dir)
        self.labeled_path = os.path.join(self.file_directory, vars.file_config.labeled_dataset_dir)
        self.detector_name = detector_name
        self.threshold, self.config = recognizer_config

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

    @abstractclassmethod
    def compare_faces(self, labeled_face_encoded_img, unlabeled_face_encoded_img, threshold):
        pass
    
    @abstractclassmethod
    def encoder(self, img, face_locations, re_sample, model):
        pass

    @abstractclassmethod
    def detection(self, detector_name):
        pass
    
    def output_img_handler(self, unlabeled_face_img, fl):
        image = unlabeled_face_img.copy()
        for i in range(len(self.unlabeled_face_encoded_img)):
            if self.best_match_names[i] != 'Unknown':
                print(f'The correct identity for Face({i+1}) is {self.best_match_names[i]} with confidance: {(self.best_match_confidences[i])*100: .2f}%')  # Print the person's name
            else:
                print(f'The identity for Face({i+1}) is Unknown')
            image = cv.putText(
                image,
                self.best_match_names[i]+f",{self.best_match_confidences[i]*100: .2f}%",  # Use the person's name
                (fl[i][0]-80, fl[i][1]-30),
                cv.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 0, 0), 
                4,
                cv.LINE_AA
            )
            toolbox.img().draw_borders(image, [fl[i]])

        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        resized_img = cv.resize(img, (int(1920*(5/7)), int(1020*(5/7))), interpolation = cv.INTER_AREA)
        toolbox.img().plot(resized_img, 'Image')

    def run(self):
        t1 = time.perf_counter()
        fl, self.face_locations = self.detection(self.detector_name)
        self.unlabeled_face_encoded_img, unlabeled_face_img = self.img_encoding(self.unlabeled_path, vars.file_config.input_img_name, self.config, self.face_locations)
        # Assuming unlabeled_face_encoded_img and fl are defined
        self.best_match_confidences = [0] * len(self.unlabeled_face_encoded_img)
        self.best_match_names = ['Unknown'] * len(self.unlabeled_face_encoded_img)
        t2 = time.perf_counter()
        # Loop over each person's profile
        for person in os.listdir(self.labeled_path):
            person_path = os.path.join(self.labeled_path, person, 'cropped_img')
            encoded_images = []
            if not os.path.exists(f'{os.path.join(person_path, person)}_encoded.pkl'):
                for labeled_face_name in os.listdir(person_path):
                    labeled_face_encoded_img, _ = self.img_encoding(person_path, labeled_face_name, self.config, full_img=True)
                    encoded_images.append(labeled_face_encoded_img)
                with open(f'{os.path.join(person_path, person)}_encoded.pkl', 'wb') as f:
                    pickle.dump(encoded_images, f)
            else:
                print(f"File {person}_encoded.pkl already exists.")
                
            with open(f'{os.path.join(person_path, person)}_encoded.pkl', 'rb') as f:
                # print(f'{os.path.join(person_path, person)}_encoded.pkl')
                encoded_images = pickle.load(f)
                # print(encoded_images)
            for encoding in encoded_images:
                for i in range(len(self.unlabeled_face_encoded_img)):
                    matches, confidence = self.compare_faces(encoding, self.unlabeled_face_encoded_img[i], self.threshold)
                    print(f'{fl[i]} -> Face({i+1}) <-> {person} <-> {matches} with confidence: {(confidence[0])*100: .2f}%')
                    if matches[0] and (confidence[0]) > self.best_match_confidences[i]:
                        self.best_match_confidences[i] = confidence[0]
                        self.best_match_names[i] = person
        t3 = time.perf_counter()
        print(f'Detection Time: {(t2-t1):.3f} s')
        print(f'Recognition Time: {(t3-t2):.3f} s')
        self.output_img_handler(unlabeled_face_img, fl)
        

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