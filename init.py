import json
import os
import vars

def read_json(file):
    with open(file, "r") as jsonfile:
        config = json.load(jsonfile)
        print("Config file read successfully")
        jsonfile.close()
    return config

def write_json(data, file):
    with open(file, "w") as jsonfile:
        json.dump(data, jsonfile)
        print("Write successfull")

def init():
    config = read_json("config.json")

    # Print the current loaded configuration
    for key in config["FaceDetectorConfig"]:
        print(key, f'\t', config["FaceDetectorConfig"][key])

    # CV2
    vars.detector_config.cv2[0]    = config["FaceDetectorConfig"]["cv2"]["scaleFactor"]
    vars.detector_config.cv2[1]    = config["FaceDetectorConfig"]["cv2"]["minNeighbors"]
    vars.detector_config.cv2[2][0] = config["FaceDetectorConfig"]["cv2"]["minLength"]
    vars.detector_config.cv2[2][1] = config["FaceDetectorConfig"]["cv2"]["minWidth"]

    # DSFD
    vars.detector_config.fd_dsfd[0] = config["FaceDetectorConfig"]["DSFD"]["detector"]
    vars.detector_config.fd_dsfd[1] = config["FaceDetectorConfig"]["DSFD"]["confidenceThreshold"]
    vars.detector_config.fd_dsfd[2] = config["FaceDetectorConfig"]["DSFD"]["nmsThreshold"]

    # DLIB
    vars.detector_config.fr_dlib[0] = config["FaceDetectorConfig"]["Dlib"]["upsampling"]
    vars.detector_config.fr_dlib[1] = config["FaceDetectorConfig"]["Dlib"]["model"]

    # MTCNN
    vars.detector_config.mtcnn[0]    = config["FaceDetectorConfig"]["MTCNN"]["minFaceSize"]
    vars.detector_config.mtcnn[1][0] = config["FaceDetectorConfig"]["MTCNN"]["thresholds"][0]
    vars.detector_config.mtcnn[1][1] = config["FaceDetectorConfig"]["MTCNN"]["thresholds"][1]
    vars.detector_config.mtcnn[1][2] = config["FaceDetectorConfig"]["MTCNN"]["thresholds"][2]
    vars.detector_config.mtcnn[2]    = config["FaceDetectorConfig"]["MTCNN"]["scaleFactor"]

    # RetinaFace
    vars.detector_config.retinaface[0] = config["FaceDetectorConfig"]["RetinaFace"]["threshold"]
    vars.detector_config.retinaface[1] = config["FaceDetectorConfig"]["RetinaFace"]["upsampleScale"] in [1]

    # RetinaNetMobileNetV1
    vars.detector_config.fd_RetinaNetMobileNetV1[0] = config["FaceDetectorConfig"]["RetinaNetMobileNetV1"]["detector"]
    vars.detector_config.fd_RetinaNetMobileNetV1[1] = config["FaceDetectorConfig"]["RetinaNetMobileNetV1"]["confidenceThreshold"]
    vars.detector_config.fd_RetinaNetMobileNetV1[2] = config["FaceDetectorConfig"]["RetinaNetMobileNetV1"]["nmsThreshold"]

    # RetinaNetResNet50
    vars.detector_config.fd_RetinaNetResNet50[0] = config["FaceDetectorConfig"]["RetinaNetResNet50"]["detector"]
    vars.detector_config.fd_RetinaNetResNet50[1] = config["FaceDetectorConfig"]["RetinaNetResNet50"]["confidenceThreshold"]
    vars.detector_config.fd_RetinaNetResNet50[2] = config["FaceDetectorConfig"]["RetinaNetResNet50"]["nmsThreshold"]

    # YOLOv8
    vars.detector_config.yolo8 = config["FaceDetectorConfig"]["YOLOv8"]["confidenceThreshold"]
    #vars.detector_config.yolo8[0] = tuple(vars.detector_config.yolo8[0])

    ####################################################################################################################
    #
    # Handling Config

    vars.handling_config.conf[0] = config["HandlingConfig"]["generatePlot"] in [1]
    vars.handling_config.conf[1] = config["HandlingConfig"]["generateOutput"] in [1]
    vars.handling_config.conf[2] = config["HandlingConfig"]["urlActive"] in [1]

    ####################################################################################################################
    #
    # Directory Config
    
    vars.file_config.input_imgs_dir       = config["DirectoryConfig"]["InputDirectory"]
    vars.file_config.output_imgs_dir      = config["DirectoryConfig"]["OutputDirectory"]
    vars.file_config.labeled_dataset_dir  = config["DirectoryConfig"]["LabeledDirectory"]
    vars.file_config.input_img_name       = config["DirectoryConfig"]["InputImgName"]
    vars.file_config.input_img_path       = os.path.join(vars.file_config.input_imgs_dir, vars.file_config.input_img_name)



