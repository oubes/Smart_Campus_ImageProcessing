import toolbox
import os
import time
from datetime import datetime
from abc import ABC, abstractclassmethod
import numpy as np

datetime_filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
file_directory = os.path.dirname(os.path.abspath(__file__))

class face_detector(ABC):
    """An abstract class for face detection models."""
    def __init__(self, in_img_path, out_path, handling_config, detector_config, img_url):
        """Initialize the face detector with the input and output paths,
        the handling and detector configurations, and the image URL."""
        self.in_img_path = os.path.join(file_directory, in_img_path)
        self.out_path = os.path.join(file_directory, out_path)

        self.img_url = img_url

        self.plot_active = handling_config[0]
        self.out_gen = handling_config[1]
        self.url_active = handling_config[2]

        self.detector_config = detector_config

    def _read_img(self):
        """Read the image from the URL or the local path and return the color and grayscale versions."""
        if self.url_active is True:
            downloaded_img_name = toolbox.url_img(self.img_url, datetime_filename).download()
            rgb_img, gray_img = toolbox.img().read(os.path.join(file_directory, downloaded_img_name), gray=True)
            toolbox.img().remove(downloaded_img_name)
        else:
            rgb_img, gray_img = toolbox.img().read(self.in_img_path, gray=True)
        return rgb_img, gray_img

    def _generate_output(self, rgb_img: np.ndarray, face_locations: list, detector_name: str):
        """Generate the output images and plots based on the face locations, the detector name, and the style."""
        if self.out_gen is True:
            current_detected_faces_dir = os.path.join('detected_faces', datetime_filename, detector_name)
            toolbox.dir(self.out_path, current_detected_faces_dir).create()
            current_output_imgs_dir = os.path.join('output_imgs', datetime_filename)
            toolbox.dir(self.out_path, current_output_imgs_dir).create()

            toolbox.img().crop_imgs(rgb_img, os.path.join(self.out_path, current_detected_faces_dir, 'face'), face_locations)
            toolbox.img().draw_borders(rgb_img, face_locations)
            toolbox.img().create(os.path.join(self.out_path, current_output_imgs_dir, f'{detector_name}'), rgb_img)

        if self.plot_active is True:
            toolbox.img().draw_borders(rgb_img, face_locations)
            toolbox.img().plot(rgb_img, detector_name)
    
    @abstractclassmethod
    def detector(self, gray_img: np.ndarray, detector_config: tuple, img: np.ndarray):
        """An abstract method for detecting faces in the grayscale image using the detector configuration."""
        pass
    
    def run(self):
        """A handler method that reads the image, detects the faces, measures the execution time, and generates the output."""
        rgb_img, gray_img = self._read_img()

        t1 = time.perf_counter()
        face_locations, detector_name, detector_config = self.detector(gray_img, self.detector_config, rgb_img)
        t2 = time.perf_counter()

        taken_time = t2 - t1
        faces_count = len(face_locations)
        
        msg = f"Detector: '{detector_name}' with config: {detector_config} has taken: {taken_time:.3f} s to detect {faces_count} faces"
        print(msg)

        self._generate_output(rgb_img, face_locations, detector_name)

        toolbox.logger().add(f'{msg}')

        return face_locations, faces_count, taken_time
    

def Detect(detector_name):
    import vars
    from Detectors import YOLOv8, DLIB, CV2, MTCNN, Retinaface, FD

    detectors = {
        'YOLOv8': (YOLOv8.yolo8_model, vars.detector_config.yolo8),
        'DLIB': (DLIB.fr_dlib_model, vars.detector_config.fr_dlib),
        'CV2': (CV2.cv2_model, vars.detector_config.cv2),
        'MTCNN': (MTCNN.mtcnn_model, vars.detector_config.mtcnn),
        'Retinaface': (Retinaface.retinaface_model, vars.detector_config.retinaface),
        # 'DSFDDetector': (FD.fd_model, vars.detector_config.fd_dsfd),
        # 'RetinaNetMobileNetV1': (FD.fd_model, vars.detector_config.fd_RetinaNetMobileNetV1),
        # 'RetinaNetResNet50': (FD.fd_model, vars.detector_config.fd_RetinaNetResNet50)
    }

    if detector_name in detectors:
        model_class, detector_config = detectors[detector_name]
        model = model_class(
            in_img_path = vars.file_config.input_img_path,
            out_path = vars.file_config.output_imgs_dir,
            handling_config = vars.handling_config.conf,
            detector_config = detector_config,
            img_url = vars.file_config.input_img_url
        )
        face_locations, faces_count, taken_time = model.run()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    # Log the results
    toolbox.logger().footer()

    return face_locations, faces_count, taken_time   



    

        


    

    