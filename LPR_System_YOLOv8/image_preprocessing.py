import cv2, imutils
import scipy.ndimage
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

def _crop_img(img: np.ndarray, box: list, style='xyxy') -> np.ndarray:
    """Crop the image according to the bounding box.

    Parameters:
    img (np.ndarray): The image array to crop.
    box (list): The coordinates of the bounding box.
    style (str): The style of the bounding box, either 'xyxy' or 'xywh'.

    Returns:
    crop_img (np.ndarray): The cropped image array.
    """
    try:
        if (style == 'xyxy') & (box is not None):
            x_min, y_min, x_max, y_max = map(int, box)
            crop_img = img[y_min:y_max, x_min:x_max]
            return crop_img
    
    except IndexError:
        raise ValueError('Cropping Failed due to empty bounding box')

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

def quality_enhancement(img: np.ndarray, upsample: int):
    img_denoise = cv2.fastNlMeansDenoising(img) 
    img_upscale = scipy.ndimage.zoom(img_denoise, upsample, order=5)
    return img_upscale

def lp_alignment(lps_img: np.ndarray) -> np.ndarray:
    """
    Aligns the license plate images for better recognition by the easyocr reader.

    Parameters:
        lps_img (np.ndarray): An array of license plate images.

    Returns:
        np.ndarray: An array of aligned license plate numbers.
    """
    edge_detection = cv2.Canny(lps_img, 0, 200)

    contours, _ = cv2.findContours(edge_detection.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lp_contour = None 
    lp_area = 0

    total_area = lps_img.shape[0] * lps_img.shape[1]

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 4: 
            if not cv2.isContourConvex(approx):
                hull = cv2.convexHull(approx)
                area = cv2.contourArea(hull)
            else:
                area = cv2.contourArea(approx)

            if area > lp_area and area >= total_area * 0.3:
                lp_area = area
                lp_contour = approx

    if lp_contour is not None:
        rect = cv2.minAreaRect(lp_contour)
        if rect[1][0] < rect[1][1]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = lps_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(lps_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
    
def preprocessing(lps_imgs: list, upsample, test_mode: bool) -> list:
    """enhance the license plate images quality for more efficient recognition using the easyocr reader.

    Parameters:
    lp_imgs (list): The list of image arrays of the license plates.

    Returns:
    lps_imgs_enhanced (list): The list of license plate numbers.
    """
    lps_imgs_enhanced = []
    for img in lps_imgs:
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_enhanced = quality_enhancement(gray_img, upsample)
            img_alignment = lp_alignment(img_enhanced)
            img_binary = cv2.adaptiveThreshold(img_alignment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 0.5) 
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) 
            img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel) 
            lps_imgs_enhanced.append(img_morph)
            
            # Start Testing
            if test_mode:
                img_vars = [gray_img, img_enhanced, img_alignment, img_binary, img_morph]
                img_title = ['Original image', 'Upsampled image', 'Aligned image', 'Binarized image', 'Morphological image']
                img_pos = range(231, 232+len(img_vars))
                plt.figure(figsize=(15, 7))
                for pos, var, title in zip(img_pos, img_vars, img_title):
                    plt.subplot(pos); plt.imshow(var, cmap='gray'); plt.title(title)
                plt.show()
            # End Testing
            
    return lps_imgs_enhanced
