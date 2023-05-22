import cv2
import numpy as np

# Read image
im = cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE)

### IMAGE PREPROCESSING ###

# Blur image to reduce noise
# blurred = cv2.GaussianBlur(im, (9, 9), 0)

# Sharpening Image
kernel = np.array([[-1, -1, -1],
                   [-1, 9,-1],
                   [-1, -1, -1]])
image_sharp = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)

# Binary Threshold Filtering
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

# Perform opening to remove noise and separate nuclei
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel2)

preproc = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR) #Converts to equal dimension
# cv2.imshow('Binary', thresh1)
# cv2.waitKey()
# cv2.destroyAllWindows()


'''
# Threshold image to binary using Otsu's method


#_, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Threshold image to binary using Adaptive Thresholding
binary = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, .7)
'''

# # Perform opening to remove noise and separate nuclei
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# preproc = opened # Save preprocessed image
# preproc = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR) #Converts to equal dimension

### BLOB DETECTOR PARAMETERS ###
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01


# Create detector with specified parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
  detector = cv2.SimpleBlobDetector(params)
else : 
  detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(preproc) # Change this to binary or opened, depending on your choice

# Draw detected blobs as red circles
im_with_keypoints = cv2.drawKeypoints(preproc, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show images
im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
combined = np.concatenate((im_bgr, preproc, im_with_keypoints), axis = 1)
cv2.imshow("Images:", combined)
cv2.waitKey(0)
cv2.imshow("Original:", im)
cv2.waitKey(0)
cv2.imshow("Preprocessed:", preproc)
cv2.waitKey(0)
cv2.imshow("End:", im_with_keypoints)
cv2.waitKey(0)


cv2.destroyAllWindows()

