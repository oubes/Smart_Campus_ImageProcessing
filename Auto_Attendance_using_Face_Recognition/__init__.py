__version__ = '2.00.01'

from main import *
from toolbox import dir, logger, url_img
from detection import face_detector
from Detectors import YOLOv8, DLIB, CV2, MTCNN, Retinaface, FD
from recognition_refactored import face_recognizer

__all__ = (
    '__version__',
    'dir',
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
