Description:

The LPR system is an application of computer vision that uses Optical Character Recognition (OCR) on images to read vehicle registration plates. The system is designed to extract, recognize, and read the alphanumeric characters on license plates from different types of vehicles.

---------------------------------------------------------------------------------------

LPR system works as follows:

1- Initialization: The LPR model is initialized with an image and a list of languages. It also loads two YOLO models - one for detecting cars (coco_model) and another for detecting license plates (lpd_model).

2- Image Reading: The read_img method reads the image using OpenCV and converts it from BGR to RGB format.

3- Car Detection: The detect_cars method uses the coco_model to detect cars in the image. It returns the bounding boxes of the detected cars.

4- Image Cropping: The crop_imgs and crop_multi_imgs methods are used to crop multiple images based on the provided bounding boxes.

5- License Plate Detection: The detect_lps method uses the lpd_model to detect license plates in the cropped car images. It returns the bounding boxes of the detected license plates.

6- License Plate Recognition: The recognize_lps method uses EasyOCR to recognize and read the text on the license plates. It prints the position, text, and confidence score of the recognized text.

7- Execution: The run method executes all the above steps in sequence.
