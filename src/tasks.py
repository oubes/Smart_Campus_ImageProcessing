import utils.toolbox as toolbox
import importlib
from src.vars import read_json


def Detect(detector_name, img_url):
    config = read_json("config/face_rec_config.json")

    detectors = {
        "YOLOv8": (
            "Detectors.YOLOv8",
            "yolo8_model",
            config["DetectorConfig"]["YOLOv8"],
        ),
        "DLIB": ("Detectors.DLIB", "fr_dlib_model", config["DetectorConfig"]["DLIB"]),
        "CV2": ("Detectors.CV2", "cv2_model", config["DetectorConfig"]["CV2"]),
        "MTCNN": ("Detectors.MTCNN", "mtcnn_model", config["DetectorConfig"]["MTCNN"]),
        "RetinaFace": (
            "Detectors.Retinaface",
            "retinaface_model",
            config["DetectorConfig"]["RetinaFace"],
        ),
    }

    if detector_name in detectors:
        module_name, model_name, detector_config = detectors[detector_name]
        DetectorModule = importlib.import_module(module_name)
        model_class = getattr(DetectorModule, model_name)
        model = model_class(detector_config=detector_config, img_url=img_url)
        face_locations, faces_count, rgb_img = model.run()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Log the results
    toolbox.logger().footer()

    return face_locations, faces_count, rgb_img
