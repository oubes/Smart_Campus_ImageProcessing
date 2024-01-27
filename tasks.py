import toolbox

from datetime import datetime
import os

datetime_filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
file_directory = os.path.dirname(os.path.abspath(__file__))

def Detect(detector_name):
    from vars import read_json
    config = read_json('config.json')
    from Detectors import YOLOv8, DLIB, CV2, MTCNN, Retinaface #, FD

    detectors = {
        'YOLOv8': (YOLOv8.yolo8_model, config["DetectorConfig"]["YOLOv8"]),
        'DLIB': (DLIB.fr_dlib_model, config["DetectorConfig"]["DLIB"]),
        'CV2': (CV2.cv2_model, config["DetectorConfig"]["CV2"]),
        'MTCNN': (MTCNN.mtcnn_model, config["DetectorConfig"]["MTCNN"]),
        'RetinaFace': (Retinaface.retinaface_model, config["DetectorConfig"]["RetinaFace"]),
        # 'DSFDDetector': (FD.fd_model, vars.detector_config.fd_dsfd),
        # 'RetinaNetMobileNetV1': (FD.fd_model, vars.detector_config.fd_RetinaNetMobileNetV1),
        # 'RetinaNetResNet50': (FD.fd_model, vars.detector_config.fd_RetinaNetResNet50)
    }

    if detector_name in detectors:
        model_class, detector_config = detectors[detector_name]
        model = model_class(
            detector_config = detector_config,
            img_url = config["ImgConfig"]["InputImgUrl"]
        )
        face_locations, faces_count, rgb_img = model.run()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Log the results
    toolbox.logger().footer()

    return face_locations, faces_count, rgb_img 

def Recognize(detector_name, recognizer_name):
    from vars import read_json
    config = read_json('config.json')
    from Recognizers import DLIB
    # OpenFace, FaceNet, DeepFace, ArcFace

    recognizers = {
        'DLIB': (DLIB.fr_dlib_model, config["RecognizerConfig"]["DLIB"])
        # 'OpenFace': (OpenFace.openface_model, vars.recognizer_config.openface),
        # 'FaceNet': (FaceNet.facenet_model, vars.recognizer_config.facenet),
        # 'DeepFace': (DeepFace.deepface_model, vars.recognizer_config.deepface),
        # 'ArcFace': (ArcFace.arcface_model, vars.recognizer_config.arcface)
    }

    if recognizer_name in recognizers:
        model_class, recognizer_config = recognizers[recognizer_name]
        model = model_class(
            detector_name = detector_name,
            recognizer_config = config["RecognizerConfig"]["DLIB"]
        )
        names = model.run()
    else:
        raise ValueError(f"Unknown recognizer: {recognizer_name}")

    return names
