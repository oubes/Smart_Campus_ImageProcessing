from recognition_refactored import face_recognizer
import face_recognition
from tasks import Detect
import toolbox

class fr_dlib_model(face_recognizer):
    def compare_faces(self, labeled_face_encoded_img, unlabeled_face_encoded_imgs, threshold):
        confidence = 1 - face_recognition.face_distance(labeled_face_encoded_img, unlabeled_face_encoded_imgs)
        matches = list(confidence >= threshold)
        return matches, confidence
    
    def encoder(self, image, face_locations, config):
        re_sample = config["RecognizerConfig"]["resample"]
        model = config["RecognizerConfig"]["encodingModel"]
        img = toolbox.read(image)
        face_encoded_img = face_recognition.face_encodings(img, face_locations, re_sample, model) #dlib
        return face_encoded_img
    
    def detection(self, detector_name, img_url):
        fl, _, rgb_img = Detect(detector_name, img_url) # x1 y1 x2 y2
        face_locations = toolbox.points2rotation_format(fl) # y1, x2, y2, x1
        return fl, face_locations, rgb_img
