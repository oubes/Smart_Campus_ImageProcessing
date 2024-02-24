__version__ = '2.00.01'

from main import *
from vars import detector_config, handling_config
from toolbox import dir, img, logger, url_img
from detection import face_detector
from Detectors import YOLOv8, DLIB, CV2, MTCNN, Retinaface, FD
from recognition import face_recognizer

__all__ = (
    '__version__',
    'detector_config',
    'handling_config',
    'dir',
    'img',
    'logger',
    'url_img',
    'face_detector',
    'DLIB',
    'FD',
    'CV2',
    'Retinaface',
    'MTCNN',
    'YOLOv8',
    'face_recognizer'
)