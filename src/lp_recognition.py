import easyocr
import cv2
import numpy as np

from src.vars import read_json

config = read_json("config/lp_config.json")
reader = easyocr.Reader([config["LprConfig"]["lang"]], verbose=True)
# letter_reader = easyocr.Reader(['ar'], recog_network='letter_model', verbose=True)
# number_reader = easyocr.Reader(['ar'], recog_network='number_model', verbose=True)
import tensorflow as tf
import keras


def _recognize_lp(lp_img: list, allow_list: str) -> list:
    """Recognize the license plate number in the image using the easyocr reader.

    Parameters:
    lp_img (np.ndarray): The image array of the license plate.

    Returns:
    lp_text (np.ndarray): The license plate number.
    """

    if lp_img is not None:
        # print(lp_img)
        lp_img_p1 = lp_img # [0]
        # lp_img_p2 = lp_img[1]
        lp_img_p2 = None
        # horizontal_list, free_list = reader.detect(lp_img_p1)
        # print(horizontal_list[0])
        # print(free_list[0])
        # return
        # cv2.imshow("lp_img_p1", lp_img_p1)
        # cv2.waitKey(0)
        if lp_img_p2 is None:
            result = reader.readtext(lp_img_p1, allowlist=allow_list[0] + allow_list[1])
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
        # print(lp_imgs[0].shape)
        # cv2.imshow("lp_imgs", lp_imgs[0])
        # cv2.waitKey(0)
        return [
            _recognize_lp(lp_img=lp_img, allow_list=allow_list) for lp_img in lp_imgs
        ]


def char_segmentation(img, rgb_img, count):
    height, width = img.shape
    windowWidthR = int(0.18 * width * count)
    windowWidthL = int(0.12 * width * count)
    windowWidth = windowWidthR
    if count == 2:
        windowWidthR = windowWidthL
    startRatio = int(0.1 * windowWidth)
    marginWidth = int(windowWidth * 0.01)
    stepSize = 8
    GroupSize = 8
    lowerBlackLimit = 0.07
    upperBlackLimit = 0.9

    Letters = []
    Letters_imgs = []
    Letters_imgs_thresh = []

    for i in range(0, width - (windowWidth), stepSize * GroupSize):

        Group = []
        for j in range(i, min(i + stepSize * GroupSize, width - (windowWidth)), stepSize):
            #plt.imshow(img[:, j:j+windowWidth],cmap='gray')
            #plt.show()
            if j <= width/2:
                windowWidth = windowWidthL
            else:
                windowWidth = windowWidthR

            blackCountMarginL = marginWidth * height  - np.count_nonzero(img[:, j : j+marginWidth])
            blackCountMarginR = marginWidth * height  - np.count_nonzero(img[:, j+windowWidth-marginWidth : j+windowWidth])
            blackCountInner = (windowWidth - 2 * startRatio) * height - np.count_nonzero(img[:, j+startRatio : j+windowWidth-startRatio])
 
            #print(blackCountMarginL, marginWidth * height)
            #print(blackCountMarginR, marginWidth * height)
            #print(blackCountInner, (windowWidth - 2 * marginWidth) * height)
            #and blackCountMarginR < 0.2 * marginWidth * height \

            if blackCountMarginL < 0.25 * marginWidth * height and blackCountMarginR < 0.17 * marginWidth * height and blackCountInner > lowerBlackLimit * (windowWidth - 2 * startRatio) * height \
                and blackCountInner < upperBlackLimit * (windowWidth - 2 * startRatio) * height:
                Group.append((blackCountInner, j))
        if len(Group) > 0:
            max_G = max(Group)[1]
            if len(Letters) == 0 or len(Letters) > 0 and max_G - Letters[-1] > 21:
                Letters.append(max_G)
              # rect_coord = [rgb_img[0][0],rgb_img[0][-1],max(Group)[1],max(Group)[1]+windowWidth]
              # rect_coord = np.array(rect_coord,dtype='int32')
                rgb2_img = rgb_img[:, max_G:max_G+windowWidth]
                Letters_imgs.append((rgb2_img, max_G))
                #               plt.imshow(rgb2_img)
                thresh_img = img[:, max_G:max_G+windowWidth]
                Letters_imgs_thresh.append((thresh_img, max_G))
              #plt.imshow(img[:, max(Group)[1]:max(Group)[1]+windowWidth],cmap='gray')
              #plt.show()

    return Letters_imgs,Letters_imgs_thresh



def image_to_symbols(enhanced_lp: np.ndarray) -> tuple[list, list]:
    # from src.lp_recognition import find_lp_contour
    # cnts = cv2.findContours(enhanced_lps.copy(), cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #
    # fond_lp_contour(enhanced_lps, 1)


    numbers_cropped = []
    numbers_cropped_thresh = []
    letters_cropped = []
    letters_cropped_thresh = []

    # for enhanced_lp in enhanced_lps:
    #     if np.average(enhanced_lp[1][:,0,0] < np.average(enhanced_lp[0][:,0,0])):
    #         enhanced_lp[0], enhanced_lp[1] = enhanced_lp[1], enhanced_lp[0]

    symbols_cropped, symbols_cropped_thresh = char_segmentation(enhanced_lp[0], enhanced_lp[0], 2)
    for i in range(len(symbols_cropped)):
        numbers_cropped.append(symbols_cropped[i][0])
        numbers_cropped_thresh.append(symbols_cropped_thresh[i][0])

    symbols_cropped, symbols_cropped_thresh = char_segmentation(enhanced_lp[1], enhanced_lp[1], 2)
    for i in range(len(symbols_cropped)):
        letters_cropped.append(symbols_cropped[i][0])
        letters_cropped_thresh.append(symbols_cropped_thresh[i][0])

    return numbers_cropped_thresh, letters_cropped_thresh

# code={'س':0,'و':1,'ظ':2,'ط':3,'غ':4,'ف':5,'ا':6,'٣':7,'ض':8,'ث':9,'ذ':10,'٩':11,'ق':12,'١':13,'ش':14,'٤':15,'ص':16,'ب':17,'٥':18,'ت':19,'لا':20,'٠':21,'ي':22,'ج':23,'ح':24,'خ':25,'٧':26,'ز':27,'٨':28,'ه':29,'د':30,'٢':31,'ك':32,'م':33,'ر':34,'ل':35,'ن':36,'٦':37,'ع':38 }
#
# def getname(n):
#     for k,v in code.items():
#         if v==n:
#             return k

def model_predict(img, model, labels):
    # labels = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س', 12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع', 18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل', 23: 'لا', 24: 'م', 25: 'ن', 26: 'ه', 27: 'و', 28: 'ى', 29: '٠', 30: '١', 31: '٢', 32: '٣', 33: '٤', 34: '٥', 35: '٦', 36: '٧', 37: '٨', 38: '٩'}

    # print(type(img))
    # print(img)
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img_rgb = np.repeat(img, 3, axis=-1)
    # print(img_rgb.shape)
    pred = model.predict(img_rgb)
    print(pred)
    pred = np.argmax(pred, axis=1)
    pred_characters = [labels[idx] for idx in pred]
    print(pred_characters)
    # labels = {idx: getname(idx) for idx in np.unique(pred)}
    # pred_characters = [labels[id] for idx in pred]
    return pred_characters

