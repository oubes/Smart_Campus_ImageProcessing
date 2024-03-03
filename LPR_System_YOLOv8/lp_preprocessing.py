import scipy.ndimage, cv2
import numpy as np
import matplotlib.pyplot as plt

def read_img(img) -> np.ndarray:
    """Read the image file and convert it to RGB format.

    Returns:
    rgb_img (np.ndarray): The RGB image array.
    """
    bgr_img = cv2.imread(img)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a given angle.

    Parameters:
    image (np.ndarray): The input image.
    angle (float): The angle by which to rotate the image in degrees.

    Returns:
    np.ndarray: The rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _crop_img(img: np.ndarray, box: list, style: str, type: str) -> np.ndarray:
    """Crop an image to a specified bounding box.

    Parameters:
    img (np.ndarray): The input image.
    box (list): The bounding box to which to crop the image.
    style (str, optional): The style of the bounding box. Defaults to 'xyxy'.

    Returns:
    np.ndarray: The cropped image.
    """
    try:
        if (style == 'xyxy') & (box is not None):
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_img = img[y_min:y_max, x_min:x_max]
            h, _, _ = cropped_img.shape
            # cv2.imshow('cropped img', cropped_img); cv2.waitKey(0)
            if type == 'lp':
                ratio = np.divide(300, h)
                cropped_img = cv2.resize(cropped_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            elif type == 'car' and h > 600:
                ratio = np.divide(600, h)
                cropped_img = cv2.resize(cropped_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            return cropped_img
    except IndexError:
        raise ValueError('Cropping failed due to incorrect bounding box')

def crop_imgs(imgs: list, boxes: list, style: str ='xyxy', type: str = None) -> list:
    """Crop multiple images from multiple images according to the bounding boxes.

    Parameters:
    imgs (list): The list of image arrays to crop.
    boxes (list): The list of bounding boxes.
    style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

    Returns:
    cropped_imgs (list): The list of cropped image arrays.
    """
    return [_crop_img(img=img, box=box, style=style, type=type) for img, box in zip(imgs, boxes)]

def find_lp_contour(lp_img: np.ndarray, enhance: dict) -> np.ndarray:
    """
    Finds the contour of the license plate image using edge detection and approximation.

    Parameters:
        lps_img (np.ndarray): An array of license plate images.
        enhance (dict): A dictionary of enhancement parameters.

    Returns:
        np.ndarray: An array of points that form the license plate contour, or None if no contour is found.
    """
    # cv2.imshow('lp', lp_img)
    edge_detection = cv2.Canny(lp_img, 0, 150)
    img_dilated = cv2.dilate(edge_detection, kernel= np.ones((3, 3), dtype='uint8'))
    
    # cv2.imshow('edge_detection', edge_detection)
    # cv2.imshow('img_dilated', img_dilated)

    contours, _ = cv2.findContours(img_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lp_contour = None 
    lp_area = 0

    total_area = lp_img.shape[0] * lp_img.shape[1]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        if len(approx) >= 4: 
            if not cv2.isContourConvex(approx):
                hull = cv2.convexHull(approx)
                area = cv2.contourArea(hull)
            else:
                area = cv2.contourArea(approx)

            if area > lp_area and area >= total_area * enhance['CONTOUR_AREA_RATIO_THRESHOLD']:
                lp_area = area
                lp_contour = approx
    
    return lp_contour

def rotate_lp_image(lp_img: np.ndarray, lp_rgb_img: np.ndarray, enhance: dict) -> np.ndarray:
    """Rotate a license plate image based on its contour.

    Parameters:
    lps_img (np.ndarray): The license plate image.
    enhance (dict): A dictionary of enhancement parameters.

    Returns:
    np.ndarray: The rotated license plate image.
    """
    lp_contour = find_lp_contour(lp_rgb_img, enhance)
    if lp_contour is not None:
        rect = cv2.minAreaRect(lp_contour)
        if rect[1][0] < rect[1][1]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)
        angle = rect[-1]
        # print(angle)
        if angle < -45:
            angle = 90 + angle
        if abs(angle) <= enhance['Max_Rotation_Angle']:
            img_rotated = rotate_image(lp_img, angle)
            rgb_img_rotated = rotate_image(lp_rgb_img, angle)

        else:
            img_rotated = lp_img
            rgb_img_rotated = lp_rgb_img
        
        return img_rotated, rgb_img_rotated, True
    else:
        return lp_img, lp_rgb_img, False

def quality_enhancement(img: np.ndarray, upsample: int) -> np.ndarray:
    """Enhance the quality of an image by denoising and upscaling.

    Parameters:
    img (np.ndarray): The input image.
    upsample (int): The factor by which to upscale the image.

    Returns:
    np.ndarray: The enhanced image.
    """
    img_denoise = cv2.fastNlMeansDenoising(img)
    img_upscale = scipy.ndimage.zoom(img_denoise, upsample, order=5)
    return img_upscale

def egypt_remover(img: np.ndarray, bgr_img: np.ndarray, enhance: dict) -> np.ndarray:
    h, w, _ = bgr_img.shape; hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    Range = np.array([[0, 100, 100],
                      [360, 255, 255]]); 
    mask = cv2.inRange(hsv_img, Range[0,:], Range[1,:]); # cv2.imshow('mask', mask)
    valid_check = np.divide(float(len(np.where(mask[:int(h*0.45), :] == 255)[0])), float(w*h))
    if valid_check >= 0.08:
        min_height = np.max(np.where(mask[:int(h*0.45),int(w//3):int((2*w)//3)] == 255)[0])
        new_img = img[int(min_height*enhance['upsample_before_aligment']*0.95):, :]
        return new_img
    else:
        return img
        

def preprocessing(lps_imgs: list, enhance: dict, test_mode: bool) -> list:
    """enhance the license plate images quality for more efficient recognition using the easyocr reader.

    Parameters:
    lp_imgs (list): The list of image arrays of the license plates.

    Returns:
    lps_imgs_enhanced (list): The list of license plate numbers.
    """
    lps_imgs_enhanced = []
    for img in lps_imgs:
        if enhance['EN']:
            if img is not None:
                img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                gray_img = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)
                
                img_enhanced_before_aligment = quality_enhancement(gray_img, enhance['upsample_before_aligment'])
                
                img_alignment, bgr_img_alignment, aligment_state = rotate_lp_image(img_enhanced_before_aligment, img, enhance)
                
                egypt_removed = egypt_remover(img_alignment, bgr_img_alignment, enhance)
                
                img_enhanced_after_aligment = quality_enhancement(egypt_removed, enhance['upsample_after_aligment'])
                
                img_binary = cv2.adaptiveThreshold(img_enhanced_after_aligment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, enhance['block_size'], 2) 
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (enhance['MORPH_KERNEL_SIZE'], enhance['MORPH_KERNEL_SIZE'])) 
                img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel) 
                
                h, w = img_morph.shape; ratio = np.divide(250, h)
                std_size_img = cv2.resize(img_morph, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
                                
                lps_imgs_enhanced.append(std_size_img)
                
                
                # Start Testing
                if test_mode:
                    img_vars = [
                        gray_img,
                        img_enhanced_before_aligment,
                        img_alignment if aligment_state is True else None,
                        egypt_removed,
                        img_enhanced_after_aligment,
                        img_binary,
                        img_morph,
                        std_size_img,
                        
                    ]
                    
                    img_title = [
                        'Gray Image',
                        'Upsample Before Aligment',
                        'Aligned Image' if aligment_state is True else None,
                        "Removing Egypt",
                        "Upsample After Aligment",
                        'Binarized Image',
                        'Morphological Image',
                        'Transfer Image to std size',
                        
                    ]
                    
                    img_pos = range(252, 253+len(img_vars))
                    plt.figure(figsize=(15, 7))
                    plt.subplot(251); plt.imshow(img); plt.title('Original image')
                    for pos, var, title in zip(img_pos, img_vars, img_title):
                        if var is not None:
                            plt.subplot(pos); plt.imshow(var, cmap='gray'); plt.title(title)    
                    plt.show()
                # End Testing
        else:
            # Start Testing
            for img in lps_imgs:
                if img is not None and test_mode:
                    plt.figure(figsize=(15, 7)); plt.imshow(img); plt.title('Original image'); plt.show()
            # End Testing
            
            return lps_imgs
            
    return lps_imgs_enhanced
        
    
