import json
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from vars import config
from dotenv import load_dotenv


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
def edit_config(config: Config, token: str):
    pass
    # Todo
    # change config.json


# Todo
# Do not take config and only detect


class Recognizable(BaseModel):
    img_url: str
    encoded_dict: list[
        dict[str, str | None]
    ]  # list[dict["id": id, "imgs": list[list[float]]]


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

    res = model_class.Recognize(
        unlabeled_img_url=person.img_url, encoded_dict=person.encoded_dict
    )
    return res


class Encodable(BaseModel):
    img: str
    prev: str | None


@app.post("/encoding")
def encoding(sayed: Encodable, token: str | None = None):
    if token != apiToken:
        return json_res(401, {"error": "UNAUTHORIZED", "message": "Invalid token"})

    imgs = sayed.img
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

    model_class = recognizers[recognizer]

    encoded_dict = model_class.add_labeled_encoded_entry(
        labeled_face_url=imgs, encoded_dict=encoded_dict
    )
    response = jsonable_encoder(str(encoded_dict))
    return response

    # from tasks import Detect
    # from preprocessing import save_augmented_imaged

    # responses = []
    # for i, img in enumerate(imgs):
    #    face_locations, face_counts, _ = Detect(detector, img)
    #    if face_counts == 0:
    #        return json_res(
    #            404,
    #            {
    #                "error": "BAD_REQUEST",
    #                "message": f"No face found in image '{img}'",
    #            },
    #        )
    #    if face_counts > 1:
    #        return json_res(
    #            404,
    #            {
    #                "error": "BAD_REQUEST",
    #                "message": f"Multiple faces found in image '{img}'",
    #            },
    #        )

    #    import toolbox

    #    face_locations = toolbox.points2rotation_format(face_locations)

    #    image = requests.get(img)
    #    with open("./tmp/image.jpg", "wb") as file:
    #        file.write(image.content)

    #    encoded_dict = model_class.encoder(
    #        image="./tmp/image.jpg",
    #        face_locations=face_locations,
    #        config=recognizerConfig,
    #    )
    #    responses[i] = jsonable_encoder(str(encoded_dict))
    #    os.remove("./tmp/image.jpg")

    # return responses


# Todo
# Connect to server
# Connect to database
# Change code accordingly
# Encoding to database and updating the database
