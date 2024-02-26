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

def _crop_img(img: np.ndarray, box: list, style='xyxy') -> np.ndarray:
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
            crop_img = img[y_min:y_max, x_min:x_max]
            return crop_img
    except IndexError:
        raise ValueError('Cropping failed due to incorrect bounding box')

def rotate_lp_image(lps_img: np.ndarray, enhance: dict) -> np.ndarray:
    """Rotate a license plate image based on its contour.

    Parameters:
    lps_img (np.ndarray): The license plate image.
    enhance (dict): A dictionary of enhancement parameters.

    Returns:
    np.ndarray: The rotated license plate image.
    """
    lp_contour = find_lp_contour(lps_img, enhance)
    if lp_contour is not None:
        rect = cv2.minAreaRect(lp_contour)
        if rect[1][0] < rect[1][1]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)
        angle = rect[-1]
        print(angle)
        if angle < -45:
            angle = 90 + angle
        if abs(angle) <= enhance['Max_Rotation_Angle']:
            rotated = rotate_image(lps_img, angle)
        else:
            rotated = lps_img
        return rotated, True
    else:
        return lps_img, False

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

def crop_imgs(imgs: list, boxes: list, style: str ='xyxy') -> list:
    """Crop multiple images from multiple images according to the bounding boxes.

    Parameters:
    imgs (list): The list of image arrays to crop.
    boxes (list): The list of bounding boxes.
    style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

    Returns:
    cropped_imgs (list): The list of cropped image arrays.
    """
    return [_crop_img(img=img, box=box, style=style) for img, box in zip(imgs, boxes)]

def find_lp_contour(lps_img: np.ndarray, enhance: dict) -> np.ndarray:
    """
    Finds the contour of the license plate image using edge detection and approximation.

    Parameters:
        lps_img (np.ndarray): An array of license plate images.
        enhance (dict): A dictionary of enhancement parameters.

    Returns:
        np.ndarray: An array of points that form the license plate contour, or None if no contour is found.
    """
    edge_detection = cv2.Canny(lps_img, 0, 200)

    contours, _ = cv2.findContours(edge_detection.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lp_contour = None 
    lp_area = 0

    total_area = lps_img.shape[0] * lps_img.shape[1]

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
   
def preprocessing(lps_imgs: list, enhance: dict, test_mode: bool) -> list:
    """enhance the license plate images quality for more efficient recognition using the easyocr reader.

    Parameters:
    lp_imgs (list): The list of image arrays of the license plates.

    Returns:
    lps_imgs_enhanced (list): The list of license plate numbers.
    """
    lps_imgs_enhanced = []
    for img in lps_imgs:
        if img is not None and enhance['EN']:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_enhanced_before_aligment = quality_enhancement(gray_img, enhance['upsample_before_aligment'])
            img_alignment, aligment_state = rotate_lp_image(img_enhanced_before_aligment, enhance)
            img_enhanced_after_aligment = quality_enhancement(img_alignment, enhance['upsample_after_aligment'])
            img_binary = cv2.adaptiveThreshold(img_enhanced_after_aligment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, enhance['block_size'], 2) 
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (enhance['MORPH_KERNEL_SIZE'], enhance['MORPH_KERNEL_SIZE'])) 
            img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel) 
            lps_imgs_enhanced.append(img_morph)
            
            # Start Testing
            if test_mode:
                img_vars = [gray_img, img_enhanced_before_aligment, img_alignment if aligment_state is True else None, img_enhanced_after_aligment, img_binary, img_morph]
                img_title = ['Gray Image', 'Upsample Before Aligment', 'Aligned Image' if aligment_state is True else None, "Upsample After Aligment", 'Binarized Image', 'Morphological Image']
                img_pos = range(242, 243+len(img_vars))
                plt.figure(figsize=(15, 7))
                plt.subplot(241); plt.imshow(img); plt.title('Original image')
                for pos, var, title in zip(img_pos, img_vars, img_title):
                    if var is not None:
                        plt.subplot(pos); plt.imshow(var, cmap='gray'); plt.title(title)    
                plt.show()
            # End Testing
            
            return lps_imgs_enhanced
        else:
            # Start Testing
            for img in lps_imgs:
                if img is not None and test_mode:
                    plt.figure(figsize=(15, 7)); plt.imshow(img); plt.title('Original image'); plt.show()
            # End Testing
            
            return lps_imgs
    
