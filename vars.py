import os
class file_config:
    labeled_dataset_dir = 'dataset/labeled'
    input_img_url = 'https://i.postimg.cc/9fjyyMjP/IMG-20231225-132606.jpg'
      
class detector_config:
    fr_dlib = [1, 'hog'] 
    # (up_sampling, model_type)
    # (1 -> 6, 'hog' or 'cnn')
    cv2 = [1.05, 4, (10, 10)]
    # (scale_factor, min_nh, min_win_size)
    # (1.01 -> 3, 1 -> 25, (1, 1) -> (50, 50))
    retinaface = [0.5, True] 
    # (threshold, upsampling)
    # (0.01 - 0.99, True or False)
    mtcnn = [10, [0.6, 0.7, 0.7], 0.709]
    # (min_face_size, steps_threshold, scale_factor)
    # (1 -> 50, [0.01 -> 0.99, 0.01 -> 0.99, 0.01 -> 0.99], 0.33 -> 1)
    yolo8 = (0.2) 
    # (threshold)
    # (0.01 -> 0.99)

class recognizer_config:
    fr_dlib = [(0.55), ((8, 'large'), (False))]
    # (threshold, (resample, encoding_model, encoding_update))
    # (40% -> 80%,, 1 -> 10, small/large)


class handling_config:
    Detectors = ['YOLOv8', 'DLIB', 'CV2', 'MTCNN', 'Retinaface']
    Recognizers = ['DLIB']
    detector_name = Detectors[1]
    recognizer_name = Recognizers[0]
