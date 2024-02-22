import numpy as np
import easyocr
from vars import read_json

config = read_json('config.json')
reader = easyocr.Reader([config['LprConfig']['lang']], verbose=False)

def _recognize_lp(lp_img: list, allow_list: str) -> np.ndarray:
    """Recognize the license plate number in the image using the easyocr reader.

    Parameters:
    lp_img (np.ndarray): The image array of the license plate.

    Returns:
    lp_text (np.ndarray): The license plate number.
    """

    if lp_img is not None:
        lp_img = np.array(lp_img)
        result = reader.readtext(lp_img, allowlist=allow_list)
        text = [res[1] for res in result]
        lp_text = "".join(text)
        print(f'LP Text: {lp_text}')
        return lp_text

        
def recognize_lps(lp_imgs: list, allow_list: str) -> list:
    """Recognize the license plate numbers in the images using the easyocr reader.

    Parameters:
    lp_imgs (list): The list of image arrays of the license plates.

    Returns:
    lps (list): The list of license plate numbers.
    """
    return [_recognize_lp(lp_img=lp_img, allow_list=allow_list) for lp_img in lp_imgs]