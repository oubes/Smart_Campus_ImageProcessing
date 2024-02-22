import cv2
import scipy.ndimage
import numpy as np

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

def lp_alignment(lps_img: np.ndarray) -> np.ndarray:
    """
    Aligns the license plate images for better recognition by the easyocr reader.

    Parameters:
        lps_img (np.ndarray): An array of license plate images.

    Returns:
        np.ndarray: An array of aligned license plate numbers.
    """
    return lps_img
    
def enhance(lps_imgs: list, upsample) -> list:
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
            img_upscale = scipy.ndimage.zoom(gray_img, upsample, order=3)
            img_alignment = lp_alignment(img_upscale)
            lps_imgs_enhanced.append(img_alignment)
            # plt.imshow(cv2.cvtColor(img_alignment, cv2.COLOR_BGR2RGB))
            # plt.show()
    return lps_imgs_enhanced