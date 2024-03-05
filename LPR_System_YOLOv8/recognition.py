import numpy as np
import easyocr
from vars import read_json
from lp_data_processing import process_and_structure

config = read_json('config.json')
reader = easyocr.Reader([config['LprConfig']['lang']], verbose=True)

def _recognize_lp(lp_img: list, allow_list: str) -> np.ndarray:
    """Recognize the license plate number in the image using the easyocr reader.

    Parameters:
    lp_img (np.ndarray): The image array of the license plate.

    Returns:
    lp_text (np.ndarray): The license plate number.
    """

    if lp_img is not None:
        lp_img_p1 = lp_img[0]; lp_img_p2 = lp_img[1]
        if lp_img_p2 is None:
            result = reader.readtext(lp_img_p1, allowlist=allow_list[0]+allow_list[1])
            text = [res[1] for res in result]
            lp_text = ["".join(text)]
            
        else:
            result1 = reader.readtext(lp_img_p1, allowlist=allow_list[1])
            result2 = reader.readtext(lp_img_p2, allowlist=allow_list[0])
            text1 = [res[1] for res in result1]
            text2 = [res[1] for res in result2]
            lp_text = ["".join(text1), "".join(text2)]
        return lp_text
        
def recognize_lps(lp_imgs: list, allow_list: str) -> list:
    """Recognize the license plate numbers in the images using the easyocr reader.

    Parameters:
    lp_imgs (list): The list of image arrays of the license plates.

    Returns:
    lps (list): The list of license plate numbers.
    """
    if lp_imgs is not None:
        return [_recognize_lp(lp_img=lp_img, allow_list=allow_list) for lp_img in lp_imgs]