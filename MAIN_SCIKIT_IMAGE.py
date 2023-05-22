import matplotlib.pyplot as plt
from skimage import feature, io, filters
from skimage.color import rgb2gray
import numpy as np

# Load image (Becomes Grayscale)
image = io.imread("test3.jpg", as_gray=True)

#Binary filter
thresh = filters.threshold_otsu(image)
binary = image > thresh

# Detect blobs
blobs = feature.blob_log(binary, max_sigma=30, num_sigma=10, threshold=.1)

# Print number of blobs found
print('Number of blobs found:', len(blobs))

# Create figure and axes
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Display original image
ax.imshow(image, cmap='gray')

# Draw a red circle on top of the original image for each blob detected
for blob in blobs:
    y, x, area = blob
    ax.add_patch(plt.Circle((x, y), area*np.sqrt(2), color='r', fill=False))

# Display the image with blobs highlighted

plt.title(f'Number of blobs found: {len(blobs)}')
plt.savefig("Test3Output.jpg") #save as jpg
#plt.show()
