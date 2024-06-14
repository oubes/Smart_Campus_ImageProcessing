# import tensorflow as tf
import keras
from skimage.transform import resize
import numpy as np

from src.lp_new import imageToSymbol
from src.lp_recognition import model_predict

def predict(img, model, labels):
    img = resize(img, (32, 32, 3), anti_aliasing=True)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    print(f'preds = {preds}')
    return labels[np.argmax(preds)]


def lp_predict(lp_img):
    # LETTER_LABELS = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر', 10: 'ز', 11: 'س', 12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع', 18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل', 23: 'م', 24: 'ن', 25: 'ه', 26: 'و', 27: 'ى'}
    LETTER_LABELS = ['ا', 'ب', 'ت', 'ج', 'د', 'ر', 'س', 'ص', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'و', 'ى']
    NUMBER_LABELS = {0: '٠', 1: '١', 2: '٢', 3: '٣', 4: '٤', 5: '٥', 6: '٦', 7: '٧', 8: '٨', 9: '٩'}

    try:
        letter_model = keras.models.load_model("pretrained_models/letter_model_no_normalization.h5")
        number_model = keras.models.load_model("pretrained_models/latest_number_model_epoch_53.keras")
        # arabic_model = keras.models.load_model("pretrained_models/arabic-OCR.h5")
    except FileNotFoundError:
        print("Failed to load pretrained models. Please check if the models are saved in the correct location.")
    except Exception as e:
        print(e)

    # print(letter_model.summary())
    # print(number_model.summary())
    Nums, Chars = imageToSymbol(lp_img)

    labelList = []
    for char in Chars:
        # predicting with letters' model
        # predicted_label = predict(char, model=letter_model, labels=LETTER_LABELS)

        predicted_label = model_predict(char, letter_model, LETTER_LABELS)
        labelList.append(predicted_label)

    for num in Nums:
        # predicting with numbers' model
        # predicted_label = predict(num, model=number_model, labels=NUMBER_LABELS)

        predicted_label = model_predict(num, number_model, NUMBER_LABELS)
        labelList.append(predicted_label)

    label = ''.join(str(e) for e in labelList)
    print(label)
    return label
