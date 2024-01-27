import toolbox
import vars
from datetime import datetime
import os

datetime_filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
file_directory = os.path.dirname(os.path.abspath(__file__))

def Detect(detector_name):
    from Detectors import YOLOv8, DLIB, CV2, MTCNN, Retinaface #, FD

    detectors = {
        'YOLOv8': (YOLOv8.yolo8_model, vars.detector_config.yolo8),
        'DLIB': (DLIB.fr_dlib_model, vars.detector_config.fr_dlib),
        'CV2': (CV2.cv2_model, vars.detector_config.cv2),
        'MTCNN': (MTCNN.mtcnn_model, vars.detector_config.mtcnn),
        'Retinaface': (Retinaface.retinaface_model, vars.detector_config.retinaface),
        # 'DSFDDetector': (FD.fd_model, vars.detector_config.fd_dsfd),
        # 'RetinaNetMobileNetV1': (FD.fd_model, vars.detector_config.fd_RetinaNetMobileNetV1),
        # 'RetinaNetResNet50': (FD.fd_model, vars.detector_config.fd_RetinaNetResNet50)
    }

    if detector_name in detectors:
        model_class, detector_config = detectors[detector_name]
        model = model_class(
            detector_config = detector_config,
            img_url = vars.file_config.input_img_url
        )
        face_locations, faces_count, rgb_img = model.run()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Log the results
    toolbox.logger().footer()

    return face_locations, faces_count, rgb_img 

def Recognize(detector_name, recognizer_name):
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
        names = model.run()
    else:
        raise ValueError(f"Unknown recognizer: {recognizer_name}")

    return names