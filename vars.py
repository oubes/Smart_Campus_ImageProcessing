import os


class file_config:
    input_img_name = 'IMG_20231225_132607.jpg'
    input_imgs_dir = 'input/dataset/unlabeled'
    input_img_path = os.path.join(input_imgs_dir, input_img_name)
    labeled_dataset_dir = 'input/dataset/labeled'
    input_img_url = 'https://media.istockphoto.com/id/1369917180/photo/large-group-of-college-students-listening-to-their-professor-on-a-class.jpg?s=612x612&w=0&k=20&c=VKWAFCEmSzPWf0Xx-o4uVgo2opkrhMemIxjhFuUueGE='
    output_imgs_dir = 'output'
      
class detector_config:
    fr_dlib = (1, 'hog') 
    # (up_sampling, model_type)
    # (1 -> 6, 'hog' or 'cnn')
    cv2 = (1.05, 4, (10, 10)) 
    # (scale_factor, min_nh, min_win_size)
    # (1.01 -> 3, 1 -> 25, (1, 1) -> (50, 50))
    retinaface = (0.5, True) 
    # (threshold, upsampling)
    # (0.01 - 0.99, True or False)
    mtcnn = (10, [0.6, 0.7, 0.7], 0.709) 
    # (min_face_size, steps_threshold, scale_factor)
    # (1 -> 50, [0.01 -> 0.99, 0.01 -> 0.99, 0.01 -> 0.99], 0.33 -> 1)
    yolo8 = (0.2) 
    # (threshold)
    # (0.01 -> 0.99)

class recognizer_config:
    fr_dlib = ((0.65), (3, 'large')) # (threshold, (resample, encoding_model))
    # (40% -> 80%,, 1 -> 10, small/large)

class handling_config:
    conf = (False, False, False) # (plot_active, output_gen, url_active)
