# Quiz: Applying Sobel

Write a function that will be useful for the Advanced Lane Finding Project at the end of this lesson! Your goal is to identify pixels where the gradient of an image falls within a specified threshold range.

**Example**

![thresh-x-example.png](../../images/thresh-x-example.png)

Here's the scaffolding for your function:

~~~python
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Grayscale
    # Apply cv2.Sobel()
    # Take the absolute value of the output from cv2.Sobel()
    # Scale the result to an 8-bit range (0-255)
    # Apply lower and upper thresholds
    # Create binary_output
    return binary_output
~~~

Pass in `img` and set the parameter `orient` as `'x'` or `'y'` to take either the x or y gradient. Set `thresh_min` and `thresh_max` to specify the range to select for `binary output`. You can usee exclusive (`<, >`) or inclusive (`<=, >=`) thresholding.

**NOTE**: Your output should be an array of the same size as the input image. The output array element should be `1` where gradients were in the threshold range and `0` everywhere else.

If you run into any errors as you run your code, refer to the **Examples of Useful Code** section in the **Sobel Operator** section and make sure your code syntax matches up. You can download the image used in the quiz [signs-vehicles-xygrad.png](../../images/signs-vehicles-xygrad.jpg)

~~~python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Read in an image and grayscale it
image = mpimg.imread('signs_vehicles_xygrad.png')

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=20, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(img) # Remove this line
    return binary_output
    
# Run the function
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
~~~