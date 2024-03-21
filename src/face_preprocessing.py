import cv2
import numpy as np
import os
import utils.toolbox as toolbox
from src.vars import config
from src.tasks import Detect
from datetime import datetime


def flip_image(image):
    if config["PreprocessingConfig"]["FilterEable"]["enable_flip"]:
        image = cv2.flip(image, 1)  # 1 for horizontal flip
    return image


def adjust_brightness(image):
    # # Increase brightness by 20%
    # brightness_factor = 1.2
    # image = np.clip(image * brightness_factor, 0, 255)
    #
    # # Decrease brightness by 20%
    # brightness_factor = 0.8
    # image = np.clip(image * brightness_factor, 0, 255)
    # return image

    contrast_factor = config["PreprocessingConfig"]["CONTRAST"]["lowest_contrast"]
    brightness_factor = config["PreprocessingConfig"]["BRIGHTNESS"][
        "highest_brightness"
    ]
    # Convert image to float32 for adjustment
    image_float = image.astype(np.float32) / 255.0
    # Adjust brightness and contrast
    adjusted_image = cv2.addWeighted(
        image_float, contrast_factor, image_float, 0, brightness_factor
    )
    # Clip values to be in the valid range [0, 1]
    adjusted_image = np.clip(adjusted_image, 0, 1)
    # Convert back to uint8
    adjusted_image = (adjusted_image * 255).astype(np.uint8)
    return adjusted_image


def apply_gaussian_blur(image):
    # Apply Gaussian blur with a random kernel size
    kernel_size = (
        int(
            np.random.uniform(
                config["PreprocessingConfig"]["KERNAL"]["lowest_gaussian_blur"],
                config["PreprocessingConfig"]["KERNAL"]["highest_gaussian_blur"],
            )
        )
        * 2
        + 1
    )  # Ensure odd kernel size
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def apply_color_jittering(image):
    if config["PreprocessingConfig"]["FilterEable"]["enable_color_jittering"]:
        # Randomly adjust gamma and saturation
        gamma = np.random.uniform(
            config["PreprocessingConfig"]["GAMMA"]["lowest_color_jittering"],
            config["PreprocessingConfig"]["GAMMA"]["highest_color_jittering"],
        )
        saturation = np.random.uniform(
            config["PreprocessingConfig"]["SATURATION"]["lowest_saturation"],
            config["PreprocessingConfig"]["SATURATION"]["highest_saturation"],
        )
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
    if config["PreprocessingConfig"]["FilterEable"]["enable_noise"]:
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
    if config["PreprocessingConfig"]["FilterEable"]["enable_histogram_equalization"]:
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
    # reading the image
    image = cv2.imread(image)
    augmented_images = {"image": image}

    # Randomly flip the image horizontally
    augmented_images["flipped"] = flip_image(image)
    # Randomly adjust brightness and contrast
    if config["PreprocessingConfig"]["FilterEable"]["enable_brightness"] == "True":
        augmented_images["bright"] = adjust_brightness(image)
    # Apply Gaussian blur
    if config["PreprocessingConfig"]["FilterEable"]["enable_gaussian_blur"] == "True":
        augmented_images["blur"] = apply_gaussian_blur(image)
    # Apply color jittering
    augmented_images["jitter"] = apply_color_jittering(image)
    # Apply noise injection
    augmented_images["noise"] = apply_noise(image)
    # Apply histogram equalization
    augmented_images["histogram"] = apply_histogram_equalization(image)
    # flip_Gaussian blur
    augmented_images["flipped_blur"] = apply_gaussian_blur(augmented_images["flipped"])
    # flip_brightness
    augmented_images["flipped_bright"] = adjust_brightness(augmented_images["flipped"])
    # flip_histo
    augmented_images["flipped_histo"] = apply_histogram_equalization(
        augmented_images["flipped"]
    )
    # flip_noise
    augmented_images["flipped_noise"] = apply_noise(augmented_images["flipped"])
    # flip_jittered
    augmented_images["flipped_jitter"] = apply_color_jittering(
        augmented_images["flipped"]
    )
    # brighness_noise
    augmented_images["bright_noise"] = adjust_brightness(augmented_images["noise"])
    # brighness_gaussian
    augmented_images["bright_blur"] = adjust_brightness(augmented_images["blur"])
    # gaussian_jittered
    augmented_images["blur_jitter"] = apply_gaussian_blur(augmented_images["jitter"])
    # gaussian_histo
    augmented_images["blur_histo"] = apply_gaussian_blur(augmented_images["histogram"])
    # histo_noise
    augmented_images["histo_noise"] = apply_histogram_equalization(
        augmented_images["noise"]
    )
    # jittered_noise
    augmented_images["jitter_noise"] = apply_color_jittering(augmented_images["noise"])
    # jittered_histo
    augmented_images["jitter_histo"] = apply_color_jittering(
        augmented_images["histogram"]
    )

    return augmented_images


# def encode_augmanted(img_path: str):
#     import face_recognition
#
#     face_locations, fc, _ = Detect("RetinaFace", img_path)
#     if fc > 1:
#         raise Exception("More than one face detected")
#     elif fc < 1:
#         raise Exception("No face detected")
#     img = cv2.imread(img_path)
#     re_sample = config["RecognizerConfig"]["DLIB"]["resample"]
#     model = config["RecognizerConfig"]["DLIB"]["encodingModel"]
#     face_encoded_img = face_recognition.face_encodings(
#         img, face_locations, re_sample, model
#     )
#     # print(face_encoded_img)
#     return face_encoded_img


def save_augmented_imaged(image_url):
    if not (is_single_person(image_url)):
        print("Please use a valid personal photo")
    else:
        img_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        image = toolbox.url_img(image_url, img_name).download()
        preprocessed_images = image_augmentation(image)
        toolbox.remove(img_name + ".jpg")

        # Save or use augmented images for training
        for i, augmented_image in enumerate(preprocessed_images):
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Augmented_imgs",
                datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                + "_"
                + str(i + 1)
                + ".jpg",
            )
            cv2.imwrite(output_path, augmented_image)
            encoded_pics = encoded_Augmanted(output_path)
