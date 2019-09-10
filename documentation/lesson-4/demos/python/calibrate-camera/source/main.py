import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

img = mpimg.imread('../../../../images/calibration/calibration1.jpg')
plt.imshow(img)

# I'll map the coordinates of the corners in this 2D image, which I will
# call it's image points to the 3D coordinates of the real undistorted
# chessboard corners, which I will call object points.

# Set up two empty arrays to hold object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# I'll prepare these object points by creating 6x8 points in an array each with 3 columns for the x,y,z coordinates of each corner. I'll initialize these all as zeros using numpy's zero function. 
objp = np.zeros((6*8, 3), np.float32)
# The z coordinate will stay zero, so I'll leave that as it is but for our first two columns x and y, I'll use numpy's mgrid function to generate the coordinates I want. mgrid returns the coordinate values for a given grid size. I'll shape those coordinates back into two columns, one for x and one for y.
objp[:,:,2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # x,y coordinates

for fname in images:
    # Read in calibration images of a chessboard
    # Rec: 20 imgs to get reliable calibration
    # Each chessboard 8x6 corners to detect
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # if corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
        plt.imshow(img)    

