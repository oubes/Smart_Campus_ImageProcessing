from typing import Tuple
import scipy.ndimage
import cv2
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
        if (style == "xyxy") & (box is not None):
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_img = img[y_min:y_max, x_min:x_max]
            h, _, _ = cropped_img.shape
            # cv2.imshow('cropped img', cropped_img); cv2.waitKey(0)
            if type == "lp":
                ratio = np.divide(300, h)
                cropped_img = cv2.resize(
                    cropped_img,
                    (0, 0),
                    fx=ratio,
                    fy=ratio,
                    interpolation=cv2.INTER_CUBIC,
                )
            elif type == "car" and h > 600:
                ratio = np.divide(600, h)
                cropped_img = cv2.resize(
                    cropped_img,
                    (0, 0),
                    fx=ratio,
                    fy=ratio,
                    interpolation=cv2.INTER_LINEAR,
                )
            return cropped_img
    except IndexError:
        raise ValueError("Cropping failed due to incorrect bounding box")


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
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
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def crop_imgs(imgs: list, boxes: list, style: str = "xyxy", type: str = "") -> list:
    """Crop multiple images from multiple images according to the bounding boxes.

    Parameters:
    imgs (list): The list of image arrays to crop.
    boxes (list): The list of bounding boxes.
    style (str): The style of the bounding boxes, either 'xyxy' or 'xywh'.

    Returns:
    cropped_imgs (list): The list of cropped image arrays.
    """
    return [
        _crop_img(img=img, box=box, style=style, type=type)
        for img, box in zip(imgs, boxes)
    ]


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
    img_dilated = cv2.dilate(edge_detection, kernel=np.ones((3, 3), dtype="uint8"))

    # cv2.imshow('edge_detection', edge_detection)
    # cv2.imshow('img_dilated', img_dilated)

    contours, _ = cv2.findContours(
        img_dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
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

            if (
                area > lp_area
                and area >= total_area * enhance["CONTOUR_AREA_RATIO_THRESHOLD"]
            ):
                lp_area = area
                lp_contour = approx

    return lp_contour


def rotate_lp_image(
    lp_img: np.ndarray, lp_rgb_img: np.ndarray, enhance: dict
) -> Tuple[np.ndarray, np.ndarray, bool]:
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
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) <= enhance["Max_Rotation_Angle"]:
            img_rotated = _rotate_image(lp_img, angle)
            rgb_img_rotated = _rotate_image(lp_rgb_img, angle)

        else:
            img_rotated = lp_img
            rgb_img_rotated = lp_rgb_img

        return img_rotated, rgb_img_rotated, True
    else:
        return lp_img, lp_rgb_img, False


def quality_enhancement(img: np.ndarray, enhance: list[int]) -> np.ndarray:
    """Enhance the quality of an image by denoising and upscaling.

    Parameters:
    img (np.ndarray): The input image.
    upsample (int): The factor by which to upscale the image.

    Returns:
    np.ndarray: The enhanced image.
    """
    img_denoise = cv2.fastNlMeansDenoising(img)
    img_upscale = scipy.ndimage.zoom(img_denoise, enhance[0], order=enhance[1])
    return img_upscale


def egypt_remover(img: np.ndarray, bgr_img: np.ndarray, enhance: dict) -> np.ndarray:
    """Removes the 'Egypt' text from the license plate image.

    Parameters:
    img (np.ndarray): The original license plate image.
    bgr_img (np.ndarray): The license plate image in BGR color space.
    enhance (dict): A dictionary of enhancement factors.

    Returns:
    np.ndarray: The license plate image with the 'Egypt' text removed.
    """
    h, w, _ = bgr_img.shape
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    Range = np.array([[0, 100, 100], [360, 255, 255]])
    mask = cv2.inRange(hsv_img, Range[0, :], Range[1, :])  # cv2.imshow('mask', mask)
    valid_check = np.divide(
        float(len(np.where(mask[: int(h * 0.45), :] == 255)[0])), float(w * h)
    )
    if valid_check >= 0.08:
        min_height = np.max(
            np.where(mask[: int(h * 0.45), int(w // 3) : int((2 * w) // 3)] == 255)[0]
        )
        new_img = img[int(min_height * enhance["upsample_before_aligment"] * 0.95) :, :]
        return new_img
    else:
        return img


def lp_spiltter(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits the license plate image into two parts.

    Parameters:
    img (np.ndarray): The license plate image.

    Returns:
    tuple: A tuple containing two np.ndarray objects representing the two parts of the license plate image.
    """
    try:
        h, w = img.shape
        edges = cv2.Canny(img, 30, 150, apertureSize=3)
        img_dilated_canny = cv2.dilate(edges, kernel=np.ones((1, 13), dtype="uint8"))
        lines = cv2.HoughLinesP(
            img_dilated_canny, 0.1, np.pi / 180, 15, minLineLength=130, maxLineGap=20
        )

        mask = np.zeros_like(img)

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:  # If the line is vertical
                cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 2)

        mask = cv2.dilate(mask, kernel=np.ones((3, 5), dtype="uint8"))

        left_edge = np.where(mask.any(axis=0))[0][0]
        right_edge = np.where(mask.any(axis=0))[0][-1]

        if left_edge <= 0.12 * w and right_edge >= 0.88 * w:
            cropped_mask = mask[:, left_edge:right_edge]
            cropped_img = img[:, left_edge:right_edge]

            h, w = cropped_mask.shape
            y = h // 3

            search_limit = int(w * 0.1)

            left_edge = next(
                (i for i, x in enumerate(cropped_mask[y, :search_limit]) if x == 0),
                None,
            )
            right_edge = next(
                (
                    i
                    for i, x in reversed(
                        list(enumerate(cropped_mask[y, -search_limit:]))
                    )
                    if x == 0
                ),
                None,
            )

            if right_edge is not None:
                right_edge = (
                    w - search_limit + right_edge
                )  # Adjust the right edge index

            cropped2_img = cropped_img[:, left_edge:right_edge]
            h, w = cropped2_img.shape

            img_split_p1 = cropped2_img[:, : int((w / 2) - 0.01 * w)]
            img_split_p2 = cropped2_img[:, int((w / 2) + 0.01 * w) :]

        else:
            img_split_p1 = img
            img_split_p2 = None
            cropped2_img = img

        return img_split_p1, img_split_p2
    except Exception:
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
        if enhance["EN"]:
            if img is not None:
                # Reduce background visibility and increase it for foreground
                img_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                gray_img = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2GRAY)

                # upsmaple and filtering before aligment to enhance the probabiltiy of successive aligment
                enhance_options = [
                    enhance["upsample_before_aligment"],
                    enhance["upsampling_before_aligment_order"],
                ]
                img_enhanced_before_aligment = quality_enhancement(
                    gray_img, enhance_options
                )

                # Align license plate horizontally
                img_alignment, bgr_img_alignment, aligment_state = rotate_lp_image(
                    img_enhanced_before_aligment, img, enhance
                )

                # Remove the upper part of license plate that contains egypt
                egypt_removed = egypt_remover(img_alignment, bgr_img_alignment, enhance)

                # upsample more time to increase the probability of successive ocr
                enhance_options = [
                    enhance["upsample_after_aligment"],
                    enhance["upsampling_after_aligment_order"],
                ]
                img_enhanced_after_aligment = quality_enhancement(
                    egypt_removed, enhance_options
                )

                # binaraize the image to aliminate some of the noise & separate the chars from the rest of the image
                img_binary = cv2.adaptiveThreshold(
                    img_enhanced_after_aligment,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    enhance["block_size"],
                    2,
                )

                # Standarize the size of the output using semi-std
                h, _ = img_binary.shape
                ratio = np.divide(250, h)
                std_size_img = cv2.resize(
                    img_binary,
                    (0, 0),
                    fx=ratio,
                    fy=ratio,
                    interpolation=cv2.INTER_CUBIC,
                )

                # Try to split the 2 parts of the image for better ocr
                img_split_p1, img_split_p2 = lp_spiltter(std_size_img)

                lps_imgs_enhanced.append([img_split_p1, img_split_p2])

                # Start Testing
                if test_mode:
                    img_vars = [
                        img,
                        img_bilateral,
                        gray_img,
                        img_enhanced_before_aligment,
                        img_alignment if aligment_state is True else None,
                        egypt_removed,
                        img_enhanced_after_aligment,
                        img_binary,
                        std_size_img,
                    ]

                    img_title = [
                        "Original image",
                        "Bilateral filter",
                        "Gray Image",
                        "Upsample Before Aligment",
                        "Aligned Image" if aligment_state is True else None,
                        "Removing Egypt",
                        "Upsample After Aligment",
                        "Binarized Image",
                        "Transfer Image to std size",
                    ]

                    cv2.imshow("img_split_p1", img_split_p1)
                    if img_split_p2 is not None:
                        cv2.imshow("img_split_p2", img_split_p2)

                    img_pos = range(251, 252 + len(img_vars))
                    plt.figure(figsize=(15, 7))
                    for pos, var, title in zip(img_pos, img_vars, img_title):
                        if var is not None:
                            plt.subplot(pos if pos % 10 != 0 else pos + 1)
                            plt.imshow(var, cmap="gray")
                            plt.title(title)
                    plt.show()
                # End Testing
        else:
            # Start Testing
            for img in lps_imgs:
                if img is not None and test_mode:
                    plt.figure(figsize=(15, 7))
                    plt.imshow(img)
                    plt.title("Original image")
                    plt.show()
            # End Testing

            return lps_imgs

    return lps_imgs_enhanced

def consecutive(data, mode ='first', stepsize=1):
    group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    group = [item for item in group if len(item)>0]

    if mode == 'first': result = [l[0] for l in group]
    elif mode == 'last': result = [l[-1] for l in group]
    return result

def word_segmentation(mat, separator_idx =  {'th': [1,2],'en': [3,4]}, separator_idx_list = [1,2,3,4]):
    result = []
    sep_list = []
    start_idx = 0
    sep_lang = ''
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0: mode ='first'
        else: mode ='last'
        a = consecutive( np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [ [item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]: # start lang
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]: # end lang
                if sep_lang == lang: # check if last entry if the same start lang
                    new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                    if sep_start_idx > start_idx:
                        result.append( ['', [start_idx, sep_start_idx-1] ] )
                    start_idx = sep[0]+1
                    result.append(new_sep_pair)
                sep_lang = ''# reset

    if start_idx <= len(mat)-1:
        result.append( ['', [start_idx, len(mat)-1] ] )
    return result

