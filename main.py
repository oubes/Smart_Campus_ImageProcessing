from detection import Detect
# from Recognizers import DLIB

# models = ['YOLOv8', 'DLIB', 'CV2', 'MTCNN', 'Retinaface']

if __name__ == "__main__":
    
#     for model in models:
#         face_locations, faces_count, taken_time = Detect(model)    
# from Recognizers import DLIB
# DLIB.fr_dlib_model(detector_name = 'MTCNN').run()

    from recognition import Recognize
    Detectors = ['YOLOv8', 'DLIB', 'CV2', 'MTCNN', 'Retinaface']
    Recognizers = ['DLIB']
    known_names = Recognize(detector_name = Detectors[1], recognizer_name = Recognizers[0])
    print(known_names)