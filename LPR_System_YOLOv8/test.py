import cv2
import numpy as np

def remove_surrounding_content(image, method="threshold_contour"):
  """
  Removes content surrounding the white space in an image using different methods.

  Args:
      image: The input image as a NumPy array.
      method: The method to use for removal (options: "threshold_contour",
               "connected_component", "flood_fill").

  Returns:
      The processed image with removed surrounding content.
  """

  if method == "threshold_contour":
    # Thresholding and Contour Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, 0, (255, 255, 255), -1)
    result = cv2.bitwise_and(image, image, mask=mask)
  elif method == "connected_component":
    # Connected Component Analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    labels, num = cv2.connectedComponents(thresh)
    largest_label = np.argmax(np.bincount(labels.flatten()))
    mask = np.zeros_like(image)
    mask[labels == largest_label] = 255
    result = cv2.bitwise_and(image, image, mask=mask)
  elif method == "flood_fill":
    # Flood Fill Algorithm
    seed_point = (image.shape[0] // 2, image.shape[1] // 2)  # Choose a central seed point
    mask = np.zeros_like(image)
    flags = cv2.FLOODFILL_FIXED_RANGE
    lo = (0, 0, 0)  # Lower color boundary (black)
    hi = (20, 20, 20)  # Upper color boundary (allow some tolerance)
    cv2.floodFill(image, mask, seed_point, (255, 255, 255), lo, hi, flags)
    result = cv2.bitwise_not(image)  # Invert the mask to isolate white area
  else:
    raise ValueError("Invalid method specified. Choose 'threshold_contour', 'connected_component', or 'flood_fill'.")

  return result

# Example usage
image = cv2.imread("test.png")  # Replace with your image path
processed_image = remove_surrounding_content(image)
cv2.imshow("Original Image", image)
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()