import cv2
import numpy as np
import os
import dlib
from vars import *
from tasks import Detect
from datetime import datetime
import toolbox

def flip_image(image):
    if config["PerprocessingConfig"]["FilterEable"]["enable_flip"]:
        image = cv2.flip(image, 1)  # 1 for horizontal flip
    return image

def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    if config["PerprocessingConfig"]["FilterEable"]["enable_brightness_contrast_adjustment"]:
        # Convert image to float32 for adjustment
        image_float = image.astype(np.float32) / 255.0
        # Adjust brightness and contrast
        adjusted_image = cv2.addWeighted(image_float, contrast_factor, image_float, 0, brightness_factor)
        # Clip values to be in the valid range [0, 1]
        adjusted_image = np.clip(adjusted_image, 0, 1)
        # Convert back to uint8
        adjusted_image = (adjusted_image * 255).astype(np.uint8)
        return adjusted_image
    else:
        return image

def apply_gaussian_blur(image):
    if config["PerprocessingConfig"]["FilterEable"]["enable_gaussian_blur"]:
        # Apply Gaussian blur with a random kernel size
        kernel_size = int(np.random.uniform(config["PerprocessingConfig"]["KERNAL"]["lowest_gaussian_blur"], config["PerprocessingConfig"]["KERNAL"]["highest_gaussian_blur"])) * 2 + 1  # Ensure odd kernel size
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image
    else:
        return image

def apply_color_jittering(image):
    if config["PerprocessingConfig"]["FilterEable"]["enable_color_jittering"]:
        # Randomly adjust gamma and saturation
        gamma = np.random.uniform(config["PerprocessingConfig"]["GAMMA"]["lowest_color_jittering"], config["PerprocessingConfig"]["GAMMA"]["highest_color_jittering"])
        saturation = np.random.uniform(config["PerprocessingConfig"]["SATURATION"]["lowest_saturation"], config["PerprocessingConfig"]["SATURATION"]["highest_saturation"])
        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Adjust gamma
        hsv_image[:, :, 2] = np.clip(gamma * hsv_image[:, :, 2], 0, 255)
        # Adjust saturation
        hsv_image[:, :, 1] = np.clip(saturation * hsv_image[:, :, 1], 0, 255)
        # Convert back to BGR
        jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return jittered_image
    else:
        return image

def apply_noise(image):
    if config["PerprocessingConfig"]["FilterEable"]["enable_noise"]:
        # Generate random noise
        noise = np.random.normal(loc=0, scale=10, size=image.shape)
        # Add noise to the image
        noisy_image = cv2.add(image, noise.astype(np.uint8))
        # Clip values to be in the valid range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image
    else:
        return image

def apply_histogram_equalization(image):
    if config["PerprocessingConfig"]["FilterEable"]["enable_histogram_equalization"]:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)
        # Merge the equalized channel with the original color channels
        equalized_image = cv2.merge([equalized_image, equalized_image, equalized_image])
        return equalized_image
    else:
        return image

def image_augmentation(image):
    #reading the image
    image = cv2.imread(image)
    
    # Randomly flip the image horizontally
    flipped_image = flip_image(image)
    # Randomly adjust brightness and contrast
    brightness_factor = np.random.uniform(config["PerprocessingConfig"]["BRIGHTNESS"]["lowest_brightness"], config["PerprocessingConfig"]["BRIGHTNESS"]["highest_brightness"])
    contrast_factor = np.random.uniform(config["PerprocessingConfig"]["CONTRAST"]["lowest_contrast"], config["PerprocessingConfig"]["CONTRAST"]["highest_contrast"])
    adjusted_image = adjust_brightness_contrast(image, brightness_factor, contrast_factor)
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)
    # Apply color jittering
    jittered_image = apply_color_jittering(image)
    # Apply noise injection
    noisy_image = apply_noise(image)
    # Apply histogram equalization
    histo_image = apply_histogram_equalization(image)
    #flip_Gaussian blur
    flip_Gaussian_blur = apply_gaussian_blur(flipped_image)
    #flip_brightness
    flip_brightness = adjust_brightness_contrast(flipped_image, brightness_factor, contrast_factor)
    #flip_histo
    flip_histo = apply_histogram_equalization(flipped_image)
    #flip_noise
    flip_noise = apply_noise(flipped_image)
    #flip_jittered
    flip_jittered = apply_color_jittering(flipped_image)
    #brighness_noise
    brightness_noise = adjust_brightness_contrast(noisy_image, brightness_factor, contrast_factor)
    #brighness_histo
    brightness_histo = adjust_brightness_contrast(histo_image, brightness_factor, contrast_factor)
    #brighness_gaussian
    brighness_gaussian = adjust_brightness_contrast(blurred_image, brightness_factor, contrast_factor)
    #gaussian_jittered
    gaussian_jittered = apply_gaussian_blur(jittered_image)
    #gaussian_histo
    gaussian_histo = apply_gaussian_blur(histo_image)
    #histo_noise
    histo_noise = apply_histogram_equalization(noisy_image)
    #jittered_noise
    jittered_noise = apply_color_jittering(noisy_image)
    #jittered_histo
    jittered_histo = apply_color_jittering(jittered_image)

    return  flipped_image, adjusted_image, blurred_image, jittered_image, noisy_image , histo_image, flip_Gaussian_blur, flip_brightness, flip_histo, flip_noise, flip_jittered, brightness_noise, brightness_histo, brighness_gaussian, gaussian_jittered, gaussian_histo, histo_noise, jittered_noise, jittered_histo


def encoded_Augmanted(img:str):
    import face_recognition
    face_locations,_,_=Detect('YOLOv8', img)
    img = cv2.imread(img)
    re_sample = config["RecognizerConfig"]["DLIB"]["resample"] 
    model = config["RecognizerConfig"]["DLIB"]["encodingModel"]
    face_encoded_img = face_recognition.face_encodings(img, face_locations, re_sample, model)
    #print(face_encoded_img)
    return face_encoded_img



def is_single_person(image_url):
    _, face_count, _ = Detect('YOLOv8', image_url)
    return face_count == 1

def save_augmented_imaged(image_url):
    if(not(is_single_person(image_url))):
        print('Please use a valid personal photo')
    else:
        img_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        image = toolbox.url_img(image_url, img_name).download()
        preprocessed_images = image_augmentation(image)
        toolbox.img().remove(img_name + '.jpg')

        # Save or use augmented images for training
        for i, augmented_image in enumerate(preprocessed_images):
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Augmented_imgs", datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '_' + str(i+1) + '.jpg')
            cv2.imwrite(output_path, augmented_image)
            encoded_pics = encoded_Augmanted(output_path)