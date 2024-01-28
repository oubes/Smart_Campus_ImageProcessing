from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from vars import *

models = ["DLIB", "CV2", "RetinaFace", "MTCNN", "YOLOv8"]
model_handlers = [lambda x: str(x)] * len(models) # Should be a list of functions to activate a certain model
class Config(BaseModel):
    # Should contain all <optional> & <required> config for all models
    # Each model should read the config that it requires only
    img: str = config["ImgConfig"]["InputImgUrl"]
    imgs: list
    ############
    # Detection
    # CV2
    cv2_scale: float = config["DetectorConfig"]["CV2"]["scale"]
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


@app.post("/{model}")
def detect(model: str, config: Config):
    if model not in models:
        return json(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find model '{model}'"
            }
        )

    for (i, def_model) in enumerate(models):
        if def_model != model:
            continue
        res = model_handlers[i](config)
        return res
