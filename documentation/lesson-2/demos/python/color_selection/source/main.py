# Libraries for operating on the image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in an image and print out some stats
image = mpimg.imread('../../../../images/highway.jpg')
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