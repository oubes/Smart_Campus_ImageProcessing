import requests
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from vars import config
from tasks import Recognize
import os
from dotenv import load_dotenv

host_config = config
detectors = ["DLIB", "CV2", "RetinaFace", "MTCNN", "YOLOv8"]
recognizers = ["DLIB"]

class Config(BaseModel):
    # Should contain all <optional> & <required> host_config for all models
    # Each model should read the host_config that it requires only
    img: str
    detector: str
    recognizer: str
    ############
    # Detection
    # CV2
    cv2_scale: float = host_config["DetectorConfig"]["CV2"]["scaleFactor"]
    cv2_min_neighbors: int = host_config["DetectorConfig"]["CV2"]["minNeighbors"]
    cv2_min_size: tuple = (host_config["DetectorConfig"]["CV2"]["minLength"], host_config["DetectorConfig"]["CV2"]["minWidth"])

    # DLIB
    dlib_upsample: int = host_config["DetectorConfig"]["DLIB"]["upsampling"]
    dlib_model: str = host_config["DetectorConfig"]["DLIB"]["model"]

    # RetinaFace
    retinaface_thresh: float = host_config["DetectorConfig"]["RetinaFace"]["threshold"]
    retinaface_scale: int = host_config["DetectorConfig"]["RetinaFace"]["upsampleScale"]

    # MTCNN
    mtcnn_min_face_size: int = host_config["DetectorConfig"]["MTCNN"]["minFaceSize"]
    mtcnn_thresh: list = host_config["DetectorConfig"]["MTCNN"]["thresholds"]
    mtcnn_scale: float = host_config["DetectorConfig"]["MTCNN"]["scaleFactor"]

    # YOLOv8
    yolo_conf_thres: float = host_config["DetectorConfig"]["YOLOv8"]["confidenceThreshold"]

    #############
    # Recognition
    # DLIB
    dlib_recog_thresh: float = host_config["RecognizerConfig"]["DLIB"]["threshold"]
    dlib_recog_resample: int = host_config["RecognizerConfig"]["DLIB"]["resample"]
    dlib_recog_model: str = host_config["RecognizerConfig"]["DLIB"]["encodingModel"]
    dlib_recog_encoding_update: int = host_config["RecognizerConfig"]["DLIB"]["encodingUpdate"]

app = FastAPI()

load_dotenv()
payload = json.dumps({
    "username": os.environ.get("USERNAME"),
    "password": os.environ.get("PASSWORD")
})
headers = {
    'Content-Type': 'application/json'
}
base_url = os.environ.get("BASE_URL")
response = requests.request("POST", base_url + "api/users/login", headers=headers, data=payload)
accessToken = response.json()["accessToken"]


# add authorization token to the header
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {accessToken}'
}
apiToken = "abcd"

body = json.dumps({
    "token": apiToken
})
response = requests.request("POST", base_url + "api/controllers/recognizer-token", headers=headers, data=body)
if not response.ok:
    print(response.text)
    print("Error in getting token")
    exit(1)


def json(code, res):
    return JSONResponse(
        status_code=code,
        content=jsonable_encoder(res)
        )

@app.get("/")
def index():
    return "Hello World!"



@app.post("/detect")
def detect(config: Config, token: str | None = None):
    if token != apiToken:
        return json(
            401,
            {
                "error": "UNAUTHORIZED",
                "message": "Invalid token"
            }
        )
    if config.detector not in detectors:
        return json(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find detection model '{config.detector}'"
            }
        )
    if config.recognizer not in recognizers:
        return json(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find recognition model '{config.recognizer}'"
            }
        )

    print(config)
    return json(
        200,
        []
    )

    for (i, def_detector) in enumerate(detectors):
        if def_detector != config.detector:
            continue
        res = Recognize(detector_name = config.detector, recognizer_name = config.recognizer)
        return res


# TODO
# Connect to server
# Connect to database
# Change code accordingly
# Encoding to database and updating the database

