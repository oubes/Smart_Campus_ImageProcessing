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
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from vars import *
from tasks import Recognize

detectors = ["DLIB", "CV2", "RetinaFace", "MTCNN", "YOLOv8"]
recognizers = ["DLIB"]

class Config(BaseModel):
    # Should contain all <optional> & <required> config for all models
    # Each model should read the config that it requires only
    img: str = config["ImgConfig"]["InputImgUrl"]
    imgs: list
    ############
    # Detection
    # CV2
    cv2_scale: float = config["DetectorConfig"]["CV2"]["scaleFactor"]
    cv2_min_neighbors: int = config["DetectorConfig"]["CV2"]["minNeighbors"]
    cv2_min_size: tuple = (config["DetectorConfig"]["CV2"]["minLength"], config["DetectorConfig"]["CV2"]["minWidth"])
    # DLIB
    dlib_upsample: int = config["DetectorConfig"]["DLIB"]["upsampling"]
    dlib_model: str = config["DetectorConfig"]["DLIB"]["model"]

    # RetinaFace
    retinaface_thresh: float = config["DetectorConfig"]["RetinaFace"]["threshold"]
    retinaface_scale: int = config["DetectorConfig"]["RetinaFace"]["upsampleScale"]

    # MTCNN
    mtcnn_min_face_size: int = config["DetectorConfig"]["MTCNN"]["minFaceSize"]
    mtcnn_thresh: list = config["DetectorConfig"]["MTCNN"]["thresholds"]
    mtcnn_scale: float = config["DetectorConfig"]["MTCNN"]["scaleFactor"]

    # YOLOv8
    yolo_conf_thres: float = config["DetectorConfig"]["YOLOv8"]["confidenceThreshold"]

    #############
    # Recognition
    # DLIB
    dlib_recog_thresh: float = config["RecognizerConfig"]["DLIB"]["threshold"]
    dlib_recog_resample: int = config["RecognizerConfig"]["DLIB"]["resample"]
    dlib_recog_model: str = config["RecognizerConfig"]["DLIB"]["encodingModel"]
    dlib_recog_encoding_update: int = config["RecognizerConfig"]["DLIB"]["encodingUpdate"]

app = FastAPI()

def json(code, res):
    return JSONResponse(
        status_code=code,
        content=jsonable_encoder(res)
        )

@app.post("/")
def bad_req():
    return json(400,{"error": "BAD_REQUEST", "message": "Unspecified Model"})

@app.get("/")
def index():
    return "Hello World!"


@app.post("/{detector}/{recognizer}")
def detect(detector: str, recognizer: str, config: Config):
    if detector not in detectors:
        return json(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find detection model '{detector}'"
            }
        )
    if recognizer not in recognizers:
        return json(
            404,
            {
                "error": "NOT_FOUND",
               "message": f"Couldn't find recognition model '{recognizer}'"
            }
        )


    for (i, def_detector) in enumerate(detectors):
        if def_detector != detector:
            continue
        res = Recognize(detector_name = detector, recognizer_name = recognizer)
        return res
