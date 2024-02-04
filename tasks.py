import toolbox
from datetime import datetime
import importlib

def Detect(detector_name):
    from vars import read_json
    config = read_json('config.json')

    detectors = {
        'YOLOv8': ("Detectors.YOLOv8", "yolo8_model", config["DetectorConfig"]["YOLOv8"]),
        'DLIB': ("Detectors.DLIB", "fr_dlib_model", config["DetectorConfig"]["DLIB"]),
        'CV2': ("Detectors.CV2", "cv2_model", config["DetectorConfig"]["CV2"]),
        'MTCNN': ("Detectors.MTCNN", "mtcnn_model", config["DetectorConfig"]["MTCNN"]),
        'RetinaFace': ("Detectors.Retinaface", "retinaface_model", config["DetectorConfig"]["RetinaFace"]),
    }

    if detector_name in detectors:
        module_name, model_name, detector_config = detectors[detector_name]
        DetectorModule = importlib.import_module(module_name)
        model_class = getattr(DetectorModule, model_name)
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
