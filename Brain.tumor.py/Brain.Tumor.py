import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the brain MRI image
image = cv2.imread("C:/Users/aswin/Brain_tumor_project/brain.tumor.jbg.jpg")
original = image.copy()
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use binary thresholding to highlight the tumor region
_, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

# Use morphological operations to remove small noises
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours (potential tumor boundaries)
contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Filter small areas
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 15)

# Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original MRI')
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Tumor Detected')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
