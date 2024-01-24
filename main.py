from detection import Detect

models = ['YOLOv8', 'DLIB', 'CV2', 'MTCNN', 'Retinaface', 'DSFDDetector', 'fd_RetinaNetMobileNetV1', 'RetinaNetResNet50']

if __name__ == "__main__":
    
    for model in models:
        face_locations, faces_count, taken_time = Detect(model)    

