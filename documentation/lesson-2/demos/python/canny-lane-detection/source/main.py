import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Use Canny Edge Detection to find the edges of the lane lines
# in an image of the road

# Read in an image
image = mpimg.imread('../../../../images/exit-ramp.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing/blurring
# Gaussian smoothing suppresses noise and spurious gradients by 
# averaging
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define threshold parameters for Canny
# NOTE: if you try running this code you might want to change these!
# edges is a binary image with white pixels tracing out the detected 
# edges and black everywhere else
low_threshold = 1
high_threshold = 10
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')