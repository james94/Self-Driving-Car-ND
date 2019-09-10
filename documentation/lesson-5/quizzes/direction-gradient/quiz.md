# Quiz: Direction of the Gradient

When you play around with the thresholding for the gradient magnitude in the previous exercise, you find what you might expect, namely, that it picks up the lane lines well, but with a lot of other stuff detected too. Gradient magnitude is at the heart of Canny edge detection and is why Canny works well for picking up all edges.

In the case of lane lines, we're interested only in edges of a particular orientation. So now, we will explore the direction or orientation of the gradient.

The direction of the gradient is simply the inverse tangent (arctangent) of the y gradient divided by the x gradient:

~~~python
arctan(sobely/sobelx)
~~~

Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of -(pie)/2 to (pie)/2. An orientation of 0 implies a vertical line and orientations of +/- (pie)/2 imply horizontal lines. (Note that in the quiz below, we actually utilize `np.arctan2`, which can return values between +/-; however, as we'll take the absolute value of sobelx, this restricts the values to +/- (pie)/2 as shown at wikipedia [atan](http://sen.wikipedia.org/wiki/Atan2)).

In this next exercise, you'll write a function to compute the direction of the gradient and apply a threshold. The direction of the gradient is much noisier than the gradient magnitude, but you should find that you can pick out particular features by orientation.

**Steps to take in this exercise**

1\. Fill out the function in the editor below to return a thresholded absolute value of the gradient direction. Use Boolean operators, again with exclusive (`<, >`) or inclusive (`<=, >=`) thresholds

2\. Test your function returns output similar to the example below for `sobel_kernel = 15, thresh = (0.7, 1.3)`

You can download the image used in the quiz [signs-vehicles-xygrad.png](../../images/signs-vehicles-xygrad.png).

![thresh-grad-dir-example](../../images/thresh-grad-dir-example.jpg)

Write the code:

~~~python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output
    
# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
~~~