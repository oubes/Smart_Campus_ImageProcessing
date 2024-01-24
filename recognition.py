import face_recognition
import os
import cv2 as cv
from detection import Detect
import vars
import toolbox
import time

def img_preprocessing():
    pass

def img_encoding(path, input_img_name, re_sample, model, face_locations=None, full_img=False):
    t1 = time.perf_counter()
    img = toolbox.img().read(os.path.join(path, input_img_name))
    if full_img:
        height, width, _ = img.shape
        face_locations = [(0, width, height, 0)]
    face_encoded_img = face_recognition.face_encodings(img, face_locations, re_sample, model) #dlib
    t2 = time.perf_counter()
    print(t2 - t1)
    return face_encoded_img, img

def compare_faces(labeled_face_encoded_img, unlabeled_face_encoded_img, threshold):
        matches = face_recognition.compare_faces(labeled_face_encoded_img, unlabeled_face_encoded_img, 1-threshold)
        face_distances = 1 - face_recognition.face_distance(labeled_face_encoded_img, unlabeled_face_encoded_img)
        return matches, face_distances

file_directory = os.path.dirname(os.path.abspath(__file__))
unlabeled_path = os.path.join(file_directory, vars.file_config.input_imgs_dir)
labeled_path = os.path.join(file_directory, vars.file_config.labeled_dataset_dir)

fl, _, _ = Detect('MTCNN') # x1 y1 x2 y2
face_locations = toolbox.img().points2rotation_format(fl) # y1, x2, y2, x1

unlabeled_face_encoded_img, unlabeled_face_img = img_encoding(unlabeled_path, vars.file_config.input_img_name, 3, 'large', face_locations)

# Assuming unlabeled_face_encoded_img and fl are defined
best_match_confidences = [0] * len(unlabeled_face_encoded_img)
best_match_names = [None] * len(unlabeled_face_encoded_img)


# Loop over each person's profile
for person in os.listdir(labeled_path):
    person_path = os.path.join(labeled_path, person, 'cropped_img')
    for labeled_face_name in os.listdir(person_path):
        labeled_face_encoded_img, _ = img_encoding(person_path, labeled_face_name, 3, 'large', full_img=True)
        
        for i in range(len(unlabeled_face_encoded_img)):
            matches, face_distances = compare_faces(labeled_face_encoded_img, unlabeled_face_encoded_img[i], 0.55)
            
            print(f'{fl[i]} -> Face({i+1}) <-> {person} <-> {labeled_face_name} -> {matches} with confidence: {(face_distances[0])*100: .2f}%')
            if matches[0] and (face_distances[0]) > best_match_confidences[i]:
                best_match_confidences[i] = face_distances[0]
                best_match_names[i] = person



image = unlabeled_face_img.copy()
for i in range(len(unlabeled_face_encoded_img)):
    if best_match_names[i] is not None:
        print(f'The correct identity for Face({i+1}) is {best_match_names[i]} with confidance: {(best_match_confidences[i])*100: .2f}%')  # Print the person's name
        image = cv.putText(
            image,
            best_match_names[i]+f",{best_match_confidences[i]*100: .2f}%",  # Use the person's name
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


# preprocessing (imp = 3/5: Quality)
# pre-encoding for labeled imgs [1 file -> person] (imp = 1/5: speed)
# code optimization and enhancing [main -> func(), config file:  -> Detector: , Generalization: Orgnization]
# Adding more models for recognition
# Connection with server
# Project discussion slides