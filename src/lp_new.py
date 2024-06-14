import cv2
import imutils
import numpy as np

def imageThresholdBinary(imgOriginal):
    '''
    Converts an image to grayscale and applies thresholding to the blurred grayscale image and saturated image
    and returns the grayscale image and a binary combination of the two thresholded images
    I/P: imgOriginal: np.ndarray
    O/P: imgGrayscale: np.ndarray, combinedThresholdImgs: np.ndarray
    '''
    imgSaturation, imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5,5), 0)

    imgThreshValue = cv2.threshold(imgBlurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    imgThreshSaturation = cv2.threshold(imgSaturation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    return imgGrayscale, (imgThreshValue & (255 - imgThreshSaturation))


def extractValue(imgOriginal):
    height, width, _ = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    _, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgSaturation, imgValue


def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    return cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)



def contourApprox(imgOriginal, cnts):
    # initialize a contour that corresponds to the receipt outline
    receiptCnt = []
    firstContour = False
    firstArea = 0
    firstPeri = 0

	# loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True) + 0.01
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        area = cv2.contourArea(approx) + 0.01

        drawnContour = imgOriginal.copy()
        cv2.drawContours(drawnContour, [approx], -1, (0, 255, 0), 2)

        if area < 3000:
            break

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        if len(approx) == 4 and (not firstContour or (firstPeri / peri < 1.4 and firstPeri / peri > 0.7 and firstArea / area < 1.4 and firstArea / area > 0.7)):
            receiptCnt.append(approx)
            if firstContour:
                break
            firstContour = True
            firstArea = area
            firstPeri = peri
        else:
            approx = cv2.convexHull(approx)
            peri = cv2.arcLength(approx, True) + 0.01
            area = cv2.contourArea(approx) + 0.01
            if len(approx) == 4 and (not firstContour or (firstPeri / peri < 1.4 and firstPeri / peri > 0.7 and firstArea / area < 1.4 and firstArea / area > 0.7)):
                receiptCnt.append(approx)
                if firstContour:
                    break
                firstContour = True
                firstArea = area
                firstPeri = peri
                break
    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
    return receiptCnt


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def showApprox(imgOriginal, thresh, receiptCnt):
    cv2.drawContours(imgOriginal, [receiptCnt], -1, (0, 255, 0), 2)
    rect_coor = np.array(receiptCnt.reshape(4,2))
    img = four_point_transform(imgOriginal, rect_coor)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    _, imgGrayscale = extractValue(img)
    imgBlurred = cv2.GaussianBlur(imgGrayscale, (5,5), 0)
    imgThreshValue = cv2.threshold(imgBlurred, 0, 255, cv2.THRESH_OTSU)[1]

    cv2.imshow("imgThreshValue", imgThreshValue)
    cv2.waitKey(0)

    return (imgThreshValue, img)


def sideBorder(img, d):
    h,w=img.shape[0:2]
    base_size=h+20,w+20,3
    if d == 1:
        base_size=h+20,w+20
    # make a 3 channel image for base which is slightly larger than target img
    base=np.zeros(base_size,dtype=np.uint8)
    if d == 1:
        cv2.rectangle(base,(0,0),(w+20,h+20),255,20) # really thick white rectangle
    else:
        cv2.rectangle(base,(0,0),(w+20,h+20),(255,255,255),20) # really thick white rectangle
    
    base[10:h+10,10:w+10]=img # this works
    return base


def CutLetters(img,rgb_img, count):
    height, width = img.shape
    windowWidthR = int(0.18 * width * count)
    windowWidthL = int(0.12 * width * count)
    windowWidth = windowWidthR
    if count == 2:
        windowWidthR = windowWidthL
    startRatio = int(0.1 * windowWidth)
    marginWidth = int(windowWidth * 0.05)
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
            if j <= width/2:
                windowWidth = windowWidthL
            else:
                windowWidth = windowWidthR

            blackCountMarginL = marginWidth * height  - np.count_nonzero(img[:, j : j+marginWidth])
            blackCountMarginR = marginWidth * height  - np.count_nonzero(img[:, j+windowWidth-marginWidth : j+windowWidth])
            blackCountInner = (windowWidth - 2 * startRatio) * height - np.count_nonzero(img[:, j+startRatio : j+windowWidth-startRatio])

            if blackCountMarginL < 0.25 * marginWidth * height and blackCountMarginR < 0.17 * marginWidth * height and blackCountInner > lowerBlackLimit * (windowWidth - 2 * startRatio) * height \
                and blackCountInner < upperBlackLimit * (windowWidth - 2 * startRatio) * height:
                Group.append((blackCountInner, j))
        if len(Group) > 0:
            max_G = max(Group)[1]
            if len(Letters) == 0 or len(Letters) > 0 and max_G - Letters[-1] > 21:
                Letters.append(max_G)
                rgb2_img = rgb_img[:, max_G:max_G+windowWidth]
                Letters_imgs.append((rgb2_img, max_G))
                thresh_img = img[:, max_G:max_G+windowWidth]
                Letters_imgs_thresh.append((thresh_img, max_G))

    return Letters_imgs,Letters_imgs_thresh

def preprocess(imgOriginal):
    imgOriginal = imutils.resize(imgOriginal, width=500)

    _ ,thresh = imageThresholdBinary(imgOriginal)

    OpenedThresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    cv2.imshow("OpenedThresh", OpenedThresh)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)

    minValue = np.min(OpenedThresh)

    OpenedThresh[imgOriginal.shape[0]-4:imgOriginal.shape[0]-1, :] = minValue
    OpenedThresh[:, imgOriginal.shape[1]-4:imgOriginal.shape[1]-1] = minValue
    OpenedThresh[:, 0:3] = minValue

    cnts = cv2.findContours(OpenedThresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    receiptCnt = contourApprox(imgOriginal, cnts)
    if len(receiptCnt) == 0:
        receiptCnt = [np.array([[[0,  0]], [[0, imgOriginal.shape[0]]],
         [[imgOriginal.shape[1], imgOriginal.shape[0]]], [[imgOriginal.shape[1], 0]]])]

    thresh = OpenedThresh
    return thresh, receiptCnt


def imageToSymbol(imgOriginal):
    thresh, receiptCnt = preprocess(imgOriginal)

    numbers_cropped = []
    numbers_cropped_thresh = []
    letters_cropped = []
    letters_cropped_thresh = []

    if len(receiptCnt) == 2:
        avg1 = np.average(receiptCnt[0][:,0,0])
        avg2 = np.average(receiptCnt[1][:,0,0])
        if avg2 < avg1:
            receiptCnt[0],receiptCnt[1] = receiptCnt[1],receiptCnt[0] 

        final_img1 ,rgb_img1 = showApprox(imgOriginal, thresh, receiptCnt[0])
        final_img1 = sideBorder(final_img1, 1)
        rgb_img1 = sideBorder(rgb_img1, 3)

        cv2.imshow("Numbers before cutting", np.concatenate((final_img1, rgb_img1), axis=0))
        cv2.waitKey(0)

        symbols_cropped, symbols_cropped_thresh = CutLetters(final_img1,rgb_img1, 2)
        for i in range(len(symbols_cropped)):
            numbers_cropped.append(symbols_cropped[i][0])
            numbers_cropped_thresh.append(symbols_cropped_thresh[i][0])
            cv2.imshow(f"Number {i}", symbols_cropped[i][0])
            cv2.waitKey(0)

        final_img1, rgb_img1 = showApprox(imgOriginal, thresh, receiptCnt[1])
        final_img1 = sideBorder(final_img1, 1)
        rgb_img1 = sideBorder(rgb_img1, 3)

        cv2.imshow("Letters before cutting", np.concatenate((final_img1, rgb_img1), axis=0))
        cv2.waitKey(0)

        symbols_cropped, symbols_cropped_thresh = CutLetters(final_img1,rgb_img1, 2)
        for i in range(len(symbols_cropped)):
            letters_cropped.append(symbols_cropped[i][0])
            letters_cropped_thresh.append(symbols_cropped_thresh[i][0])
            cv2.imshow(f"Letter {i}", symbols_cropped[i][0])
            cv2.waitKey(0)
    else:
        avg = np.average(receiptCnt[0][:,0,0])
        cv2.imshow("imgOriginal", imgOriginal)
        cv2.waitKey(0)
        final_img1, rgb_img1 = showApprox(imgOriginal, thresh, receiptCnt[0])
        final_img1 = sideBorder(final_img1, 1)
        rgb_img1 = sideBorder(rgb_img1, 3)

        cv2.imshow("len(receiptCnt) != 2, final_img1", final_img1)
        cv2.waitKey(0)
        cv2.imshow("rgb_img1", rgb_img1)
        cv2.waitKey(0)

        if final_img1.shape[1] < 280:
            if avg < final_img1.shape[1] / 2:
                symbols_cropped, symbols_cropped_thresh = CutLetters(final_img1,rgb_img1, 2)
                for i in range(len(symbols_cropped)):
                    numbers_cropped.append(symbols_cropped[i][0])
                    numbers_cropped_thresh.append(symbols_cropped_thresh[i][0])
                    cv2.imshow(f"Number {i}", symbols_cropped[i][0])
                    cv2.waitKey(0)
            else:
                symbols_cropped, symbols_cropped_thresh = CutLetters(final_img1,rgb_img1, 2)
                for i in range(len(symbols_cropped)):
                    letters_cropped.append(symbols_cropped[i][0])
                    letters_cropped_thresh.append(symbols_cropped_thresh[i][0])
                    cv2.imshow(f"Letter {i}", symbols_cropped[i][0])
                    cv2.waitKey(0)
        else: 
            symbols_cropped, symbols_cropped_thresh = CutLetters(final_img1,rgb_img1, 1)
            for i in range(len(symbols_cropped)):
                if symbols_cropped[i][1] < final_img1.shape[1] / 2:
                    numbers_cropped.append(symbols_cropped[i][0])
                    numbers_cropped_thresh.append(symbols_cropped_thresh[i][0])
                    cv2.imshow(f"Number {i}", symbols_cropped[i][0])
                    cv2.waitKey(0)
                else:
                    letters_cropped.append(symbols_cropped[i][0])
                    letters_cropped_thresh.append(symbols_cropped_thresh[i][0])
                    cv2.imshow(f"Letter {i}", symbols_cropped[i][0])
                    cv2.waitKey(0)
    return numbers_cropped_thresh, letters_cropped_thresh
