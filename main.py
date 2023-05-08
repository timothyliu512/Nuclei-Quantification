import cv2
import numpy as np

#Read image
im = cv2.imread("DAPI2.tif", cv2.IMREAD_GRAYSCALE)

### IMAGE PREPROCESSING ###

# Blur image to reduce noise
blurred = cv2.GaussianBlur(im, (5, 5), 0)

# Enhance contrast
equalized = cv2.equalizeHist(blurred)

# Threshold image to binary using Otsu's method
_, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform opening to remove noise and separate nuclei
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

preproc = opened # Save preprocessed image
preproc = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR) #Converts to equal dimension
 
### BLOB DETECTOR PARAMETERS ###
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 20;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 900
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.001
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.001
 

# Create detector with specified parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
  detector = cv2.SimpleBlobDetector(params)
else : 
  detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs
keypoints = detector.detect(binary) # Change this to binary or opened, depending on your choice
 
# Draw detected blobs as red circles
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show images
im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
combined = np.concatenate((im_bgr, preproc, im_with_keypoints), axis = 1)
cv2.imshow("Images:", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
