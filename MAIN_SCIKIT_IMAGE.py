import matplotlib.pyplot as plt
from skimage import feature, io, filters
from skimage.color import rgb2gray
import numpy as np
import cv2

# Load image (Becomes Grayscale)
FILENAME = "test1.jpg"
FILEOUTPUT = "Test1Output.jpg"

#################################
# PREPROCESSING IMAGE #
#################################

#Read in image
image = io.imread(FILENAME, as_gray=True)

#Binary ThresholdFiltering
thresh = filters.threshold_otsu(image)
binary = image > thresh

# Detect blobs (nuclei)
blobs = feature.blob_log(binary, max_sigma=5, num_sigma=10, threshold=.1)

# Print number of blobs(nuclei) found
# print('# of Nuclei:', len(blobs))
# Number of Nuclei: len(blobs)

#################################
# PLOTTING #
#################################

# Change Parameters
HEIGHT = 10
WIDTH = 10
DPI = 200

fig1, ax1 = plt.subplots(figsize=(HEIGHT, WIDTH), dpi=DPI)

ax1.imshow(image, cmap='gray')

# Draw circles on top of original image for each blob(nuclei) detected
for blob in blobs:
    y, x, area = blob
    ax1.add_patch(plt.Circle((x, y), area*np.sqrt(2), color='r', fill=False))

plt.show()

# Display the image with blobs(nuclei) highlighted
# plt.title(f'Number of Nuclei found: {len(blobs)}')
# plt.savefig(FILEOUTPUT)                 #save as jpg


# Histogram
fig2, ax2 = plt.subplots(figsize=(HEIGHT, WIDTH), dpi=DPI)

# Histogram of pixel intensity
ax2.hist(image.ravel(), bins=256, color='gray')
ax2.set_title('Histogram of Pixel Intensities')

plt.show()

print("Success!")

