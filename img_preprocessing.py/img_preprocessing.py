import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = cv2.imread("C:/Users/aswin/Downloads/img.preprocessing1.png")
    cv2.imshow('Original', image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray)

    # Remove noise with Gaussian Blur
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Blurred', blurred)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('Canny Edges', edges)

    # Simple binary threshold
    _, binary_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Threshold', binary_thresh)

    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imshow('Adaptive Threshold', adaptive_thresh)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    dilation = cv2.dilate(binary_thresh, kernel, iterations=1)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if True:
    img_path = "C:/Users/aswin/Downloads/testimg.jpg"  # Replace with your image path
    preprocess_image(img_path)

