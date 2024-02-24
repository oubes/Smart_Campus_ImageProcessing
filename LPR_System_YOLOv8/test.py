# %%
# Import libraries
import cv2
import numpy as np

# Load the image
img = cv2.imread("test2.png")

# Split the RGB channels
b, g, r = cv2.split(img)

print(b.shape)

# Threshold the blue channel
threshold = 200
mask = cv2.threshold(b, threshold, 255, cv2.THRESH_BINARY)[1]

# Find the bounding box of the blue part
x, y, w, h = cv2.boundingRect(mask)

# Draw a black rectangle over the image
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

# Show the modified image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
