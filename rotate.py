# https://stackoverflow.com/questions/72887400/install-gdal-on-linux-ubuntu-20-04-4lts-for-python
from osgeo import gdal
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Open the source dataset
src_ds = gdal.Open(

)
if src_ds is None:
    print("Unable to open the dataset.")
    sys.exit(1)

# Get the first raster band
rb = src_ds.GetRasterBand(1)
img_array = rb.ReadAsArray()

# Check if the image is in the correct format for equalization
if img_array.dtype != np.uint8:
    # Normalize and convert to uint8
    img_normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_normalized.astype(np.uint8)
else:
    img_uint8 = img_array

# Apply histogram equalization
equ = cv2.equalizeHist(img_uint8)

# Display the original and equalized images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(equ, cmap="gray")
plt.axis("off")

# rotated image
rotated_image = cv2.rotate(equ, cv2.ROTATE_90_COUNTERCLOCKWISE)
plt.subplot(1, 2, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image, cmap="gray")
plt.axis("off")

plt.show()
