from detection import Detect
from init import init

models = ['YOLOv8', 'DLIB', 'CV2', 'MTCNN', 'Retinaface']

if __name__ == "__main__":
    init()
    for model in models:
        face_locations, faces_count, taken_time = Detect(model)
