from ultralytics import YOLO
import numpy as np

coco_model = YOLO("pretrained_models/yolov8s.pt")
lpd_model = YOLO("pretrained_models/license_plate_detector.pt")


def detect_cars(img: np.ndarray, vehicles: list) -> np.ndarray:
    """Detect cars in the image using the YOLO model.

    Parameters:
    img (np.ndarray): The image array to detect cars.

    Returns:
    car_boxes (np.ndarray): The list of bounding boxes for cars.
    """
    coco_results = coco_model(img, verbose=False)
    all_boxes = np.array(coco_results[0].boxes.xyxy.cpu())
    all_labels = np.array(coco_results[0].boxes.cls.cpu())
    desired_labels = np.array(vehicles)
    idx = np.where(np.isin(all_labels, desired_labels))[0]
    return all_boxes[idx]


def _detect_lp(img: np.ndarray) -> np.ndarray:
    """Detect the license plate in the image using the YOLO model.

    Parameters:
    img (np.ndarray): The image array to detect the license plate.

    Returns:
    lp_box (np.ndarray): The bounding box for the license plate.
    """
    try:
        lp_results = lpd_model(img, verbose=False)
        lp_box = lp_results[0].boxes.xyxy[0]
        lp_box = lp_box.detach().cpu().numpy().astype(np.int16)
        return lp_box
    except IndexError:
        return None


def detect_lps(imgs: list) -> list:
    """Detect the license plates in the images using the YOLO model.

    Parameters:
    imgs (list): The list of image arrays to detect the license plates.

    Returns:
    lps_box (np.ndarray): The list of bounding boxes for the license plates.
    """
    return [_detect_lp(img=img) for img in imgs]
