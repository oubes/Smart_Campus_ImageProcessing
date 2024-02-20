from deepface import DeepFace
import os
import vars
from detection import Detect, datetime_filename

file_directory = os.path.dirname(os.path.abspath(__file__))
unlabeled_path = os.path.join(file_directory, vars.file_config.input_imgs_dir)
labeled_path = os.path.join(file_directory, vars.file_config.labeled_dataset_dir)

# define a similarity threshold, e.g. 0.4
threshold = 0.4

Detect('MTCNN')
input_img_path = os.path.join(file_directory, vars.file_config.output_imgs_dir, 'detected_faces', datetime_filename, 'mtcnn')

# Initialize an empty dictionary to store the face, person and their minimum cosine value
faces_cosine = {}

for face in os.listdir(input_img_path):
    min_cosine = float('inf')  # Initialize with infinity
    correct_person = None
    for person in os.listdir(labeled_path):
        person_path = os.path.join(labeled_path, person, 'cropped_img')

        result = DeepFace.find(
            img_path = os.path.join(input_img_path, face),
            db_path  = person_path,
            enforce_detection=False,
        )
        print(result)
        for item in result:
            VGG_Face_cosine = item.get('VGG-Face_cosine', None)
            if VGG_Face_cosine is not None:
                VGG_Face_cosine = list(VGG_Face_cosine.values)
                if VGG_Face_cosine != [] and VGG_Face_cosine[0] < min_cosine:
                    min_cosine = VGG_Face_cosine[0]
                    correct_person = person

    # Store the face, person and their minimum cosine value in the dictionary
    faces_cosine[face] = (correct_person, min_cosine)

# Print the faces and their corresponding person and minimum cosine values
for face, (person, cosine) in faces_cosine.items():
    print(f'The face {face} corresponds to the person {person} with a minimum cosine similarity score of {1-cosine}%')
