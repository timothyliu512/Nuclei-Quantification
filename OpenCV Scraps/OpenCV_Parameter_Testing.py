import cv2
import numpy as np

# Read image
im = cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE)

# IMAGE PREPROCESSING

kernel = np.array([[-1, -1, -1],
                   [-1, 9,-1],
                   [-1, -1, -1]])
image_sharp = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)

#Binary Threshold Filtering
ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

preproc = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR) #Converts to equal dimension

# Perform opening to remove noise and separate nuclei
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(preproc, cv2.MORPH_OPEN, kernel)

# Save preprocessed image
preproc = opened

#preproc = cv2.cvtColor(preproc, cv2.COLOR_GRAY2BGR)  # Converts to equal dimension

# Loop over different minConvexity values
minInertia_values = np.arange(0.01, 1.1, 0.05)  # Change this to the values you want to try

for minInertia in minInertia_values:

    # BLOB DETECTOR PARAMETERS
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 2000

    # Filter by Area
    params.filterByArea = False
    params.minArea = 10
    params.maxArea = 1000

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.01

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.71  # Set minConvexity to current value

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = minInertia

    # Create detector with specified parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(opened)

    # Draw detected blobs as red circles
    im_with_keypoints = cv2.drawKeypoints(preproc, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show images
    im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate((preproc, im_with_keypoints), axis=1)
    cv2.imshow(f"Images with minConvexity={minInertia}:", combined)

    cv2.waitKey(0)

cv2.destroyAllWindows()
