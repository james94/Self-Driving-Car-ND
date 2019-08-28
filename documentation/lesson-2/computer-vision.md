# Computer Vision Fundamentals

## Setting up the Problem

How do you teach the car how to drive itself?

You will need to teach the car how to perceive the world around it. When humans drive, they use their eyes to see how fast to go and where the lane lines are where to turn. A car does not have eyes, but in self-driving cars we can use cameras and other sensors to achieve a similar function.

What are those cameras seeing as one drives down the road?

A human sees where the lane lines are automatically, but we need to teach the car how to do that. To find lane lines, you will have to write code to identify and track the position of the lane lines in a series of images. 

Which of the features in a highway image could be useful in the identification of lane lines on the road?

Color, shape, orientation and position in the image.

## Color Selection

We will find the lane lines in an image using color. The lane lines are white.

How does one select the white pixels in an image?

To select a color, first think about what color means in the case of digital images. In some cases, it means our image is made up of a stack of three images with one each for red, green and blue. These images are sometimes called color channels. Each of these color channels contains pixels whose values range from zero to 255 where zero is the darkest possible value and 255 is the brightest possible value.

If zero is dark and 255 is bright, what color would represent pure white in our red, blue and green image?

Pure White = [R,G,B] = [255,255,255]

## Color Selection Code Example

Here is some example code for a simple color selection in Python for the following highway image:

![highway](images/highway.jpg)

~~~python
# Libraries for operating on the image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in an image and print out some stats
image = mpimg.imread('images/highway.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy of arrays or other variables
# If instead, "a = b", then all changes made to "a" will be reflected in "b" too!
color_select = np.copy(image)

# Define color selection criteria
# Note: if you run the code, these values are not sensible
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Select any pixels below the threshold and set them to zero
# Note: all pixels above the color threshold will be retained while those pixels below the threshold will be blacked out in the color_select image
threshold = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
plt.show()
~~~

Since all thresholds are set to 0, all pixels will be included in the selection. 

## Quiz: Color Selection

Enter the [Color Selection Quiz](quizzes/color_selection/quiz.md)

## Region Masking

Region masking is about focusing on the region of the image that is of interest. For a self-driving car, one can assume the camera that took the image is mounted in a fixed position on front of the car. Therefore, the lane lines will always appear in the same general region of the image.

Next, a criteration will be added to only consider pixels for color selection in the region where we expect to find lane lines.

Here is example code for doing region masking for the following highway image:

![highway](images/highway.jpg)

~~~python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('images/highway.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Pull the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Define a triangle region of interest
# left_bottom, right_bottom, apex represent the vertices of a triangular region that would be retained for the color selection while masking everything else out
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# Fit lines (y=Ax+B) to identify the 3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)


# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(region_select)

# uncomment if plot does not display
# plt.show()
~~~

## Color and Region Combined

Let's combine color selection and region masking to pull only the lane lines out of the image.

Here is Python example code requiring that a pixel meet both the color selection and region masking requirements to be retained:

~~~python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('images/highway.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

# Pull the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

# Define color selection criteria
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# Fit lines (y=Ax+B) to identify the 3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Mask color selection
color_select[color_thresholds] = [0,0,0]

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
# plt.show()
~~~

## Finding Lines of Any Color

## What is Computer Vision?

## Canny Edge Detection

## Canny to Detect Lane Lines

## Hough Transform to Find Lane Lines

