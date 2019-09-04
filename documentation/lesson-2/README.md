# Computer Vision Fundamentals

## What is Computer Vision?

Computer Vision works on enabling computers to see, identify and process images or the world in the same way that human vision does. We will get more in depth with two Computer Vision techniques used in Finding Lane Lines for self-driving cars. The first technique is Canny Edge Detection and the second technique is Hough Transform.

![opencv_python](images/opencv_python.png)

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

## Quiz: Color Region

Enter the [Color Region Quiz](quizzes/color_region/quiz.md)

## Finding Lines of Any Color

![highway_lines_painted](images/highway_lines_painted.png)

You are able to find lane lines. What happens is lane lines are not always the same color and lines of the same color under different lighting conditions (day, night, etc) may fail to be detected by our simple color selection.

What we will do next is use sophisticated computer vision methods to take the color selection algorithm to the next level, so it can detect lines of any color.

## Canny Edge Detection

The goal of Canny Edge Detection is to identify the boundaries of an object in an image. To perform the detection, convert the image to grayscale, then compute the gradient of the grayscale image.

In a gradient image, the brightness of each pixel corresponds to the strength of the gradient at that point. We can find edges by tracing out the pixels that follow the strongest gradients.

By identifying edges, we can more easily detect objects by their shape. What is an edge? Let's start by looking at the parameters for the following OpenCV Canny function:

~~~python
edges = cv2.Canny(gray, low_threshold, high_threshold)
~~~

I apply the Canny function to an image called gray and the output will be another image called edges. Low threshold and high threshold determine how strong the edges must be to be detected.
The algorithm will first detect strong edge (strong gradient) pixels above the `high_threshold`, reject pixels below the `low_threshold`, include pixels with values between `low_threshold` and `high_threshold` as long as they are connected to strong edges. The output `edges` is a binary image with white pixels tracing out the detected edges and black everywhere else. Check out the [OpenCV Canny Documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html) for more details.

What would make sense as a reasonable range for these parameters?

A reasonable range for the threshold parameters would be in the tens to hundreds because the grayscale conversion left us with an 8-bit image. Each pixel can take 2^8 = 256 values. So, the pixel values range from 0 to 255. This range implies that derivatives (the value differences from pixel to pixel) will be on the scale of tens or hundreds.

A ratio of `low_threshold` to `high_threshold` recommended by [John Canny](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps) is a low to high ratio of 1:2 or 1:3.

`cv2.Canny()` also applies Gaussian smoothing internally. Gaussian smoothing is a way of suppressing noise and spurious gradients by averaging (check out the [OpenCV documentation for GaussianBlur](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur)).

You can think of the strength of an edge as being defined by how different the values are in adjacent pixels in the image.

When looking at a grayscale image, one can see dark points, bright points and all the gray in between. Rapid changes in brightness are where we find the edges. Our image is just a mathematical function of x and y, so we can perform mathematical operations on it just like any other function.

~~~python
f(x, y) = pixel value
~~~

An example, we can take the image's derivative, which is a measure of change of the above function.

~~~python
df/dx = Triangle(pixel value)
~~~

Images are two dimensional, so it makes sense to take the derivative with respect to x and y simultaneously.

When we compute the gradient from a grayscale image, we are measuring how fast pixel values are changing at each point in an image and in which direction they're changing most rapidly. Computing the gradient gives us thick edges and with the Canny algorithm, we will thin out these edges to find the individual pixels that follow the strongest gradients. We will extend those strong edges to include pixels all the way down to a lower threshold that we defined when when calling the Canny function.

## Quiz: Canny Edge Detection

Enter the [Canny Edge Detection Quiz](quizzes/canny_edge_detection/quiz.md)

## Canny to Detect Lane Lines

You will use Canny to find the edges of the lane lines in an image of the road:

We need to read in an image:

~~~python
# Read in an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('images/exit-ramp.jpg')
plt.imshow(image)
~~~

![exit-ramp.jpg](images/exit-ramp.jpg)

**Image of the road read in**

We need to convert the image to be grayscale:

~~~python
import cv2
# grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap = 'gray')
~~~

![grayscale-exit-ramp](images/grayscale-exit-ramp.jpg)

We perform Gaussian smoothing/blurring to suppress noise and spurious gradients by averaging:

~~~python
# Define a kernel size for Gaussian smoothing/blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
~~~

We perform Canny Edge Detection get a binary image with white pixels tracing out the detected edges and black everywhere else:

~~~python
# Define threshold parameters for Canny
low_threshold = 1
high_threshold = 10
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap = 'Greys_r')
~~~

![edges-exit-ramp](images/edges-exit-ramp.jpg)

Recap, the OpenCV function `Canny` was called on a Gaussian smoothed grayscaled image called `blur_gray` and detected edges with thresholds on the gradient of `high_threshold` and `low_threshold`.

## Quiz: Canny to Detect Lane Lines

Enter the [Canny to Detect Lane Lines Quiz](quizzes/canny-lane-detection/quiz.md)

## Hough Transform

Since we used Canny Edge Detection, we now have an image full of dots that represent the edges in the original image. Now we will connect the dots in the image to look for lines. To find lines, I need to first adoppt a model of a line and then fit that model to the assortment of dots in my edge detected image.

Since the image is a function of x and y, the equation of a line `y = mx + b` can be used. In this case, my model includes two parameters: m and b.

In **Image Space**, a line is plotted as x versus y.

In **Hough Space** (also known as parameter space), that same line is represented as m versus b. 

The Hough transform is a conversion from image space to Hough space. So, the characterization of a line in image space will be a single point at the position m-b in Hough space.

## Quiz: Image Space vs Hough Space

Enter the [Image Space vs Hough Space Quiz](quizzes/image-hough-space/quiz.md)

A line in image space corresponds to a point in Hough space. What does a point in image space correspond to in Hough space?

A single point in image space has many possible lines that pass through it, but only lines with particular combinations of the m and b parametrs. By rearranging the equation of a line, we find a single point (x, y) corresponds to the line `b = y - xm`.

## Quiz: A Point in Image Space in Hough Space

Enter the [A Point in Image Space in Hough Space Quiz](quizzes/point-image-hough-space/quiz.md)

## Quiz: 2 Points in Image Space in Hough Space

Enter the [2 Points in Image Space in Hough Space Quiz](quizzes/2-points-image-hough-space/quiz.md)

## Quiz: Hough Space Intersection in Image Space

Enter the [Hough Space Intersection in Image Space Quiz](quizzes/intersection-hough-image-space/quiz.md)

Our strategy to find lines in Image Space will be to find intersecting lines in Hoough Space. We do this by dividing up our Hough Space into a grid and define intersecting lines as all lines passing through a given grid cell. To do this, we need to first run the Canny Edge Detection algorithm to find all points associated with edges in the image. I can then consider every point in this edge-detected image as a line in Hough Space. Where many lines in Hough Space intersect, I declare I have found a collection of points that describe a line in image space.

We do have a problem, vertical lines have infinite slope in m-b representation. So, we need a new parameterization. Let's redefine our line in polar coordinates. Now the variable p describes the perpendicular distance of the line from the origin. Theta is the angle of the line away from horizontal. Now each point in image space corresponds to a sine curve in Hough space. If we take a whole line of points in image space, it translates into a whole bunch of sine curves in Hough space. So, the intersection of those sine curves in theta-p space gives the parameterization of the line.

## Quiz: Square Dots in Image Space in Hough Space

Enter the [Square Dots in Image Space in Hough Space Quiz](quizzes/square-dots-image-hough-space/quiz.md)

## Hough Transform to Find Lane Lines

We are onto the implementation of Hough Transform to find lane lines. We will need to specify parameters to describe what kind of lines we want to detect (ex. long lines, short lines, bendy lines, dashed lines, etc) by using OpenCV `HoughLinesP`.

Check out [Understanding Hough Transform with Python](https://alyssaq.github.io/2014/understanding-hough-transform/) to learn how to code Hough Transform from scratch.

We are working with the following canny edge detected image:

![edges-exit-ramp.jpg](images/edges-exit-ramp.jpg)

Let's examine the input parameters for OpenCV function `HoughLinesP`:

~~~python
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
~~~

- `masked_edges` is the image we are operating on, it is the output from `Canny` OpenCV function
- `lines` is the output from `HoughLinesP`, which will be an array containing endpoints (x1, y1, x2, y2) of all line segments detected by the transform operation.

The other parameters define what kind of line segments we are looking for

- `rho` and `theta` are the distance and angular resolution of our grid in Hough Space
    - `rho` needs to be specified in units of pixels. A minimum value is 1.
    - `theta` needs to be specified in units of radians. A reasonable starting place is 1 degree (pi/180 in radians)
    - scale these two parameters to be more flexible in your definition of what constitutes a line
- `threshold` specifies the minimum number of votes (intersections in a given grid cell) a candidate line needs to have to make it into the output
- `np.array([])` is a placeholder
- `min_line_length` is the minimum length of a line (in pixels) that you will accept in the output
- `max_line_gap` is the maximum distance (in pixels) between segments tht you will allow to be connected into a single line

You can iterate through output `lines` and draw them onto the image to see what you get.


Here is the Python example code for using Hough Transform to find lane lines:

~~~python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in and grayscale the image
image = mpimg.imread('images/exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 
# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(combo)
~~~

The following image is the result:

![hough-test.jpg](images/hough-test.jpg)

As you can see in the image, a lot of line segemenets have been detected. Next is to figure out which parameters do the best job of optimizing the detection of lane lines. Earlier you learned how to apply a triangular region mask to the image to filter out parts of the image we do not need. You can also use a quadrilateral region mask using the `cv2.fillPoly()`.

## Quiz: Hough Transform

Enter the [Square Dots in Image Space in Hough Space Quiz](quizzes/hough-transform/quiz.md)

## Parameter Tuning

Parameter tuning is one of the biggest challenges in computer vision. What works well for one image may not work at all with different lighting and/or backgrounds.

Computer Vision Engineers gain an intuition over time for ranges of parameters and different techniques that might work best for a set of situations. 

A fellow Self-Driving Car student wrote the blog [Finding the right parameters for your Computer Vision algorithm](https://medium.com/@maunesh/finding-the-right-parameters-for-your-computer-vision-algorithm-d55643b6f954), which describes their approach to building a parameter tuning tool.