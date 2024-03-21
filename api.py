import json
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from src.vars import config
from dotenv import load_dotenv
from typing import Optional


host_config = config
detectors = ["DLIB", "CV2", "RetinaFace", "MTCNN", "YOLOv8"]
recognizers = ["DLIB"]


class Config(BaseModel):
    # Should contain all <optional> & <required> host_config for all models
    # Each model should read the host_config that it requires only
    detector: str = "RetinaFace"
    recognizer: str = "DLIB"
    ############
    # Detection
    # CV2
    cv2_scale: float = host_config["DetectorConfig"]["CV2"]["scaleFactor"]
    cv2_min_neighbors: int = host_config["DetectorConfig"]["CV2"]["minNeighbors"]
    cv2_min_size: tuple = (
        host_config["DetectorConfig"]["CV2"]["minLength"],
        host_config["DetectorConfig"]["CV2"]["minWidth"],
    )

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
    yolo_conf_thres: float = host_config["DetectorConfig"]["YOLOv8"][
        "confidenceThreshold"
    ]

    #############
    # Recognition
    # DLIB
    dlib_recog_thresh: float = host_config["RecognizerConfig"]["DLIB"]["threshold"]
    dlib_recog_resample: int = host_config["RecognizerConfig"]["DLIB"]["resample"]
    dlib_recog_model: str = host_config["RecognizerConfig"]["DLIB"]["encodingModel"]
    dlib_recog_encoding_update: int = host_config["RecognizerConfig"]["DLIB"][
        "encodingUpdate"
    ]


class Recognizable(BaseModel):
    img_url: str
    encoded_dict: list[
        dict[str, Optional[str]]
    ]  # list[dict["id": id, "imgs": list[list[float]]]


class Recognized(BaseModel):
    students: list[str]
    faces: int


class Encodable(BaseModel):
    img: str
    prev: Optional[str]


app = FastAPI()

load_dotenv()
payload = {
    "username": os.environ.get("USERNAME"),
    "password": os.environ.get("PASSWORD"),
}
base_url = os.environ.get("BASE_URL")
if base_url is None:
    print("BASE_URL not found")
    exit(1)
# response = requests.request("POST", base_url + "/api/users/login", json=payload)
# accessToken = response.json()["data"]["accessToken"]


# add authorization token to the header
apiToken = os.environ.get("API_TOKEN")


def json_res(code, res):
    return JSONResponse(status_code=code, content=jsonable_encoder(res))


@app.post("/config")
def edit_config(config: Config, token: Optional[str]):
    if token != apiToken:
        return json_res(401, {"error": "UNAUTHORIZED", "message": "Invalid token"})

    with open("config/face_rec_config.json") as file:
        host_config = json.load(file)

    if config.detector not in detectors:
        return json_res(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find detector '{config.detector}'",
            },
        )
    if config.recognizer not in recognizers:
        return json_res(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find recognizer '{config.recognizer}'",
            },
        )

    host_config["HandlingConfig"]["detectorName"] = config.detector
    host_config["HandlingConfig"]["recognizerName"] = config.recognizer

    host_config["DetectorConfig"]["CV2"]["scaleFactor"] = config.cv2_scale
    host_config["DetectorConfig"]["CV2"]["minNeighbors"] = config.cv2_min_neighbors
    host_config["DetectorConfig"]["CV2"]["minLength"] = config.cv2_min_size[0]
    host_config["DetectorConfig"]["CV2"]["minWidth"] = config.cv2_min_size[1]

    host_config["DetectorConfig"]["DLIB"]["upsampling"] = config.dlib_upsample
    host_config["DetectorConfig"]["DLIB"]["model"] = config.dlib_model

    host_config["DetectorConfig"]["RetinaFace"]["threshold"] = config.retinaface_thresh
    host_config["DetectorConfig"]["RetinaFace"]["upsampleScale"] = (
        config.retinaface_scale
    )

    host_config["DetectorConfig"]["MTCNN"]["minFaceSize"] = config.mtcnn_min_face_size
    host_config["DetectorConfig"]["MTCNN"]["thresholds"] = config.mtcnn_thresh
    host_config["DetectorConfig"]["MTCNN"]["scaleFactor"] = config.mtcnn_scale

    host_config["DetectorConfig"]["YOLOv8"]["confidenceThreshold"] = (
        config.yolo_conf_thres
    )

    host_config["RecognizerConfig"]["DLIB"]["threshold"] = config.dlib_recog_thresh
    host_config["RecognizerConfig"]["DLIB"]["resample"] = config.dlib_recog_resample
    host_config["RecognizerConfig"]["DLIB"]["encodingModel"] = config.dlib_recog_model
    host_config["RecognizerConfig"]["DLIB"]["encodingUpdate"] = (
        config.dlib_recog_encoding_update
    )

    with open("config/face_rec_config.json", "w") as file:
        json.dump(host_config, file, indent=4)

    return json_res(200, {"message": "Config updated successfully"})


@app.post("/recognition")
def recognition(person: Recognizable, token: str):
    if token != apiToken:
        return json_res(401, {"error": "UNAUTHORIZED", "message": "Invalid token"})
    from Recognizers import DLIB

    if person.encoded_dict is None or len(person.encoded_dict) == 0:
        return json_res(
            404,
            {
                "error": "BAD_REQUEST",
                "message": "No encoded data found in request",
            },
        )

    model_class = DLIB.fr_dlib_model(
        host_config["HandlingConfig"]["recognizerName"],
        host_config["RecognizerConfig"]["DLIB"],
    )

    res, fc = model_class.Recognize(
        unlabeled_img_url=person.img_url, encoded_dict=person.encoded_dict
    )

    response = Recognized(students=res, faces=fc)
    return jsonable_encoder(response)


@app.post("/encoding")
def encoding(sayed: Encodable, token: Optional[str] = None):
    if token != apiToken:
        return json_res(401, {"error": "UNAUTHORIZED", "message": "Invalid token"})

    img = sayed.img
    recognizer = host_config["HandlingConfig"]["recognizerName"]

    from Recognizers import DLIB

    if sayed.prev is None or sayed.prev == "":
        encoded_dict = []
    else:
        encoded_dict = list(json.loads(sayed.prev))
    recognizers = {
        "DLIB": (
            DLIB.fr_dlib_model(recognizer, host_config["RecognizerConfig"]["DLIB"])
        )
    }
    if recognizer not in recognizers:
        return json_res(
            404,
            {
                "error": "NOT_FOUND",
                "message": f"Couldn't find recognition model '{recognizer}'",
            },
        )

    from src.face_preprocessing import image_augmentation
    from utils.toolbox import download_img
    import numpy as np

    image = download_img(img, "tmp")
    augmented_imgs = image_augmentation(image)

    model_class = recognizers[recognizer]
    for enc_img in augmented_imgs.values():
        # print(enc_img)
        if (
            isinstance(enc_img, np.ndarray)
            and len(enc_img.shape) == 3
            and enc_img.shape[2] in [3, 4]
            and enc_img.dtype == np.uint8
        ):
            encoded_dict = model_class.add_labeled_encoded_entry(
                labeled_face_url=enc_img, encoded_dict=encoded_dict
            )
    response = jsonable_encoder(str(encoded_dict))
    return response
