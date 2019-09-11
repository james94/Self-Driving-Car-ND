# Advanced Lane Finding

by James Medel, April 16, 2019

## Reflection

**80% of building a self-driving car is perception**, according to Sebastian Thrun. **Computer Vision** is the art and science of understanding and **perceiving the world** around you **through images**. In the case of self-driving cars, Computer Vision helps us **detect lane markings, vehicles, pedestrians** and other elements in the environment in order to **navigate safely**. Why do we focus on the Camera images when we have self-driving cars that can employ sophisticated sensors? Humans already do a great job of driving with just 2 eyes or 1 eye and 1 brain, so we mimic it with Cameras. In addition, Cameras are less expensive than RADAR and LiDAR. There are RADAR and LiDAR sensors that can see the world in 3D, which can be a huge advantage in knowing where you are in your environment. **Cameras see in 2D**, but they have a much **higher spatial resolution** than RADAR and LiDAR, so it is **possible to infer depth information from images**. So, it is possible that self-driving cars will outfitted with a handle full of cameras and an intelligent algorithm to do the driving.

## Overview

In this project, I created a data pipeline using Python to identify lane boundaries, lane curvature and vehicle position with respect to the center of the lane for videos and images. For the Python code, check **P2.ipynb**.

`How to get started finding lane lines?`

Your ultimate goal is to measure some of the quantities that need to be known to control the car. To steer a car, you'll need to know how much your lane is curving. To do that you'll need to map out the lanes in your camera images after transforming them to a different perspective. One way of looking down on a road, but first to get this perspective transformation right, you first have to correct for the affect of image distortion. Some of the objects in the images especially the ones on the edges can get stretched or skewed in various ways and you need to correct for that.

## Camera Calibration

Purpose: to compute the transformation between 3D object points in the world and 2D image points.

### Concept

Camera Calibration removes inherent distortions known to affect its perception of the world. These distortions were handled by my **CameraCalibration** class.

#### Pinhole Camera Model vs Human Eye

`How does this image distortion occur?`

Pinhole Camera Model is a simple model of a camera.

![Pinhole Camera Model]()

![The Human Eye]()

When a camera forms an image, its looking at the world similar to how our eyes do by focusing the light that is reflected off of objects in the world.

#### 3D to 2D

![3D |> 2D]

In the above case, through a small pinhole, the camera focuses the light that's reflected off a 3D traffic sign and forms a 2D image at the back of the camera where a sensor or some film would be placed. In fact, the image it forms would be upside down and reversed because rays of light that enter from the top of an object will continue on that angled path through the pinhole and end up at the bottom of the formed image. Similarly, light that reflects off the rightside of an object will travel to the left of the formed image.

![math_3D_to_2D]()

In Math, this transformation from 3D object points `P(X,Y,Z)` to 2D image points `p(x,y)` is done by a transformative matrix called the Camera Matrix (C) `P ~ Cp`. C will be needed later on to calibrate the camera.

#### Pinhole vs Curved Lenses

![pinhole_vs_lenses]()

However, real cameras don't use tiny pinholes, they use lenses to focus multiple light rays at a time, which allows them to quickly form images, but lenses can introduce distortion too. Light rays usually bend too much or too little at the edges of a curved lense on a camera, which creates the effect we saw earlier that distorts the edges of images. 

#### Types of Distortion

![radial_distortion]()

So, lines or objects appear more or less curved than they actually are. This is called **Radial Distortion**. It is the most common type of distortion.

![tangential_distortion_father_away]()

![tangential_distortion_closer]()

Another type of distortion is tangential distortion, which occurs when the camera's lense is not aligned perfectly to the imaging plane where the camera sensor or film is, this makes the image look tilted, so some objects appear farther away or closer than they actually are.

![fisheye]()

There are also lenses that purposely distort images, such as fisheye or wide angle lenses, which keep radial distortion for a stylistic effect. Yet, for our purposes, we are using these images to position our self-driving car and eventually steer it in the right direction, so we need un-distorted images that accurately reflect our real world surroundings.

#### Distortion Coefficients and Correction

`Why are Distortion Coefficients Important?`

We can capture this image distortion by **5 Distortion Coefficients** whose values reflect the amount of radial and tangential distortion in an image. In severly distorted cases, more than 5 coefficients are required to capture the amount of distortion.

~~~python
Distortion_coefficients = (k1,k2,p1,p2,k3)
~~~

`How are distorted images undistorted?`

To undistort a particular point (x_distorted,y_distorted) in a distorted image, OpenCV calculates r. r is the distance between the equivalent point in an undistorted image (x_corrected, y_corrected) and the center of the image distortion (x_c, y_c). Also known as distortion center, which is often the center of that image (x_c, y_c).

![img_undist_one_point]()

The above picture shows how OpenCV performs distortion correction for a particular point. With OpenCV to perform image distortion correction for the entire image, we use cv2 function:

~~~python
undist_img = cv2.undistort(self.dist_img_m, mtx, dist_coeff, None, mtx)
~~~

`Radial Distortion Correction:`

~~~python
x_distorted = x_ideal(1 + k1*r^2 + k2*r^4 + k3*r^6)
y_distorted = y_ideal(1 + k1*r^2 + k2*r^4 + k3*r^6)
~~~

**k1,k2,k3** are needed for **radial distortion correction**.

`Tangential Distortion Correction:`

~~~python
x_corrected = x + [2p1*xy + p2(r^2 + 2*x^2)]
y_corrected = y + [p1(r^2 + 2*y^2) + 2*p2*xy]
~~~

**p1,p2** are needed for **tangential distortion correction**.

If we know these coefficients, we can use them to calibrate our camera and un-distort our images.

![real_world_scene]()

![distorted_image]()

![undistorted_image]()

#### Measuring Distortion

We know distortion changes the size and shape of objects in an image, so how do we calibrate for that?

Well we can take pictures of known shapes, then we can correct any known errors. We can use any shape to calibrate our camera, we'll use a chessboard. A chessboard is great for calibration because its regular high contrast pattern makes it easy to detect automatically. So, if we use our camera to take multiple pictures of a chessboard on a flat surface, then we'll be able to detect any distortion by looking at the difference between the **apparent size and shape of the squares in these images** and the **size and shape that they actually are**. Then we'll use that information to calibrate our camera, then create a transform that maps distorted points to undistorted points and finally undistort any images. 

All we need to use is Python and OpenCV.

#### Calibrating Your Camera

1\. Read in calibration images of a chessboard

Recommended: Use 20 (9x6 corners) chessboard images (at 720p resolution) to get a reliable calibration. Each image is of a chessboard taken at different angles and distances. There is a test_image to test my camera calibration and undistortion on.

2\. Finding Corners

For a chessboard image, count the number of corners in any row and enter that value for `nx`.
Then count the number of corners in a given column and store that value for `ny`.

Use OpenCV functions to auto find and draw corners in an image of a chessboard pattern.

Example of visualized image with corners drawn:

![chessboard_drawn_corners]()

- [cv2.findChessboardCorners](https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html)
- [cv2.drawChessboardCorners](https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html)


2\. Calibration steps (Background Info)

Map the coordinates of our corners in this 2D image (image points) to the 3D coordinates of the real undistorted chessboard corners (object points).

The object points will all be the same, just the known object coordinates of the chessboard corners for a 9x6 board. These points will be 3D coordinates (x,y,z) from the top left corner (0,0,0) to the bottom right (7,5,0). The z coordinate will be 0 on every point since the board is on a flat image plane. x and y  will be all the coordinates of the corners.

3\. Prepare object points from 3D to 2D points

I'll prepare and initialize these object points first by creating 9x6 points in an array each with 3 columns for the (x,y,z) coordinates of each corner. The z coordinate will stay 0, but for our first two columns x and y, I need to get the coordinate values for the given grid size (9x6) using numpy mgrid() function. I also need to shape those coordinates back into 2 columns, one for x and another for y.

4\. Create 2D image points (Finding chessboard corners)

Look at the distorted calibration[1-20].jpg image and detect the corners of the chessboard. We can use OpenCV `ret, corners = cv2.findChesssboardCorners(gray, (9,6), None)` function to detect chessboard corners that return the corners in a grayscale image. 

5\. Save object points and image points

If OpenCV function was able to find corners or image points, then we save that data to an array. We also save the prepared object points to another array. These object points will be the same for all the calibration[1-20].jpg images since they represent our real chessboard.

Optional\. Draw Detected Corners

Use OpenCV `img = cv2.drawChessboardCorners(img, (9,6), corners, ret); plt.imshow(img)` function to display detected corners or image points onto one of our real chessboard images.
Now we were able to verify corners were found for one real chessboard image.

6\. Find corners for all chessboard images

Iterate through each image file, detecting corners and appending points to the object and image points arrays. Then later, we'll use the object points and image points to calibrate this camera.

~~~python
for fname in imgs:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), corners, ret)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # save img with corners drawn
        img_corners = cv2.drawChessboardCorners(img, (9,6), corners, ret)
~~~

7\. Calibrate the Camera

We use OpenCV `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)` function to calibrate our camera, it calculates and returns the camera matrix and distortion coefficients, so we can transform 3D object points to 2D image points. It also returns the position of the camera in the world with values for rotation and translation vectors.

We can also get the image shape directly from the color image. We retrieve the first two values in the color image shape array by `1::` and reverse them by `-1`, all together `img.shape[1::-1]`.

8\. Undistort Images

We use OpenCV `dst = cv2.undistort(img, mtx, dist, None, mtx)` function correct for image distortion, it returns undistorted (known dst) image.

> Note: To calibrate your own camera, print out a [chessboard pattern](), place it on a flat surface and take 20 or 30 pictures of it. Make sure to take pictures of the pattern over the entire field of view of your camera, particularly near the edges.

### Rubric Criteria & Specification

**Rubric Criteria: How did you compute the camera matrix and the distortion coefficients?**

1. I used OpenCV, 

~~~python
ret, mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(self.m_objpoints, self.m_imgpoints, img_size, None, None), 
~~~

**Rubric: Meet Specification**

`cv2.calibrateCamera()` function to calculate the correct camera matrix and distortion coefficients using the calibration 9x6 chessboard images at 720p resolution provided in the repository.

Chessboard Image with Corner Points Drawn

![chess_corners]()

**Rubric Criteria: Provide an example of a distortion corrected calibration image**

**Rubric: Meet Specification**

The distortion matrix was used to un-distort one of the calibration images to verify that the calibration was correct.

Chessboard Image Distortion Correction Applied

![chess_undistort]()

Now we demonstrated the calibration is correct, we can use the distortion matrix to un-distort our road images.

## Pipeline: Image Processing

Used test images to verify that my pipeline works and configure parameters for video processing.

## Distortion Correction

Purpose: to ensure that the geometrical shape of objects is represented consistently, no matter where they appear in an image.

### Concept

`Why is Image Distortion Correction Important for Self-Driving Cars?`

Image Distortion is what happens when a camera looks at 3D objects in the real world and transforms them into a 2D image. This transformation isn't perfect. You can take an image of the road and look at them through different camera lenses (3 different images different distortions). You can see the edges of the lanes are bent and sort of stretched outward. Distortion changes the shape and size of these objects appear to be. This is a problem since we are trying to accurately place the self driving car in this world. Eventually we would like to look at the curve of a lane and steer in that direction. But, if the lane is distorted, we'll get the wrong measurement for curvature in the first place and our steering angle will be wrong.

`Why is it important to correct for image distortion?`

- Distortion can change the apparent size of an object in an image
- Distortion can change the apparent shape of an object in an image
- Distortion can cause an object's appearance to change depending on where it is in the field of view
- Distortion can make objects appear closer or farther away than they actually are

`What is the 1st step in Analyzing Camera Images?`

Undo this image distortion, so we can get correct and useful information out of them.

### Rubric Criteria & Specification

**Rubric Criteria: Provide an example of a distortion corrected image**

We already verified distortion correction that was calculated via camera calibration has been applied to a calibration image. Now we can apply distortion correction to each road image. 

Show an example of a distortion corrected road image:

![lane_undistort]()

Links to Types of Camera Distortion: 

- [Radial Distortion]()
- [Tangential Distortion]()

## Color & Gradient Thresholds 

Learn how to use gradient thresholds and different color spaces to more easily identify lane markings on the road.

## Concept

### Gradient Threshold

### Sobel Operator

### Magnitude of Gradient

### Direction of Gradient

### Combining Thresholds

### Color Spaces

### Color Thresholding

### HLS Intuitions

### HLS and Color Thresholds

### Color and Gradient

### Rubric Criteria & Specification

(Binary Image)

**Rubric Criteria: How did you use Color Transforms to Create a Binary Image?**

RGB for white lane line pixels

Combine RGB = (R | G) & B

HLS for yellow lane line pixels

Combine HLS = H | (S & L)

Combine Colors = RGB | HLS

**Rubric Criteria: How did you use Gradient Transforms to Create a Binary Image?**

Sobel-X

Gradient Magnitude

Gradient Direction

for edges

Combine Gradient = Sobel-X & Gradient Magnitude & Gradient Direction

**How did you combine binary images to best detect yellow and white lane line pixels?**

Gradient + Color = Combine Color | Combine Gradient

## Perspective Tranform 

Purpose: to transform an image such that we are effectively viewing objects from a different angle or direction.

### Concept

Maps the image points in a given image to different image points to give a new perspective. The bird's eye view transform is important because it let's us view a lane from above, which is useful for calculating the lane curvature later on. 

Perspective is the phenomon that objects appear smaller the farther they are from a viewpoint.

Parallel lines seem to converge to a point

Many artists use perspective to give the right impression of an object's size, depth and position viewed from a particular point. Let's look at perspective in this image of the road.

![lane_img_perspective]()

As you can see the lane appears to get smaller the farther it gets from the camera. The background scenery also appears smaller than the trees closer to the camera in the foreground. Mathematically, we can characterize perspective by saying that real world coordinates (x,y,z), the greater the magniutde of an object's z coordinate or distance from a camera, the smaller it will appear in a 2D image.

A perspective transform uses this information to transform an image. It essentially transforms the apparent z coordinate of the object points, which in turn changes that object's 2D representation. A perspective transform warps the image and effectively drags points towards or pushes them away from the camera to change the apparent perspective.

To change the road image to a bird's eye view scene, we can apply a perspective transform that zooms in on the farther away objects. This is useful because some tasks, such as finding the curvature are easier to perform on a bird's eye view of an image.

Lane Line before bird's eye view transformation:

![lane_img_prior_bev]()

We can see tthe left lane line looks like is going toward the right and the right lane line looks like it is moving toward the left. By doing a perspective transform and looking at the lanes from above, we can see that the lanes are parallel.

![show_prior_after_bev]()

Thus, we can see that the lane lines curve about the same amount to the right. So, a perspective transform lets us change the perspective, so we can view the same scene from different viewpoints and angles. This can be looking at the scene from the side of a camera, below the camera or looking down on the road from above.

![show_map_to_bev_lane]()

Doing a bird's eye view transform is helpful for road images because it will help to match a car's location directly with a map since map's display roads and scenary from a top down view.

![src_imgpts_to_dst_imgpts]()

Above image shows perspective transform mapping image points to different image points to give a birds eye view transform.

(Bird's Eye View)

### Rubric Criteria & Specification

**Rubric Criteria: How did you perform perspective tranform?**

1\. I computed the perspective transform, M, using source and destination points:

I selected 4 source points programmatically by taking the width and height shape of image and multiplying it by a certain percentage of the image that I wanted to grab.

~~~python
M = cv2.getPerspectiveTransform(src, dst)
~~~

Another approach for selecting 4 source points and 4 destination points is to use corners from when we found chessboard corners, so we can transform the image perspective.

2\. I warped an image using perspective transform, M:

~~~
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
~~~

**Rubric Criteria: Provide an example of a transformed image?**

## Identify Lane-Line Pixels 

**Rubric Criteria: How did you identify lane-line pixels?**

**Rubric Criteria: How did you fit their positions with a polynomial?**

(Fit a Polynomial)

**Search from Prior is extra**

## Radius of Lane Curvature

### Concept

Now that we have calibrated our camera and corrected images for distortion, we can extract lane curvature from the images of the road. Think about how you drive on a highway, you take a look at the road in front of you and the cars around you. You press the gas or break to go with the flow and based on how much the lane is curving left or right, you turn the steering wheel to stay in that lane.

`How does this work for a self-driving car?`

Self-driving cars need to be told the correct steering angle to turn left or right. We can calculate this angle is we know a few things about the speed, dynamics of the car and how much the lane is curving. To detect the lane line curvature, first we'll detect the lane lines using some masking and thresholding techniques, then perform a perspective transform to get a bird's eye view of the lane, which will let us fit a 2nd degree polynomial to that lane line, then we can extract the curvature of the lanes from this polynomial with just a little math.

Image below from Camera Calibration.

![lane_curvature_1]()

In the above image, we use math to extract the curvature from the polynomials.

In the above image, we are dealing with a lane line that is close to vertical, we can fit a line using this formula:

~~~python
f(y) = Ay^2 + By + C
~~~

A = the curvature of the lane line

B = the direction the line is pointing

C = the position of the line based on how far away it is from the very left of an image (y = 0)




### Rubric Criteria & Specification

**Rubric Criteria: How did you calculate the radius of lane curvature?**

### Vehicle Position in Regards to Center

**Rubric Criteria: How did you calculate the position of the vehicle with respect to center?**

### Result Plotted Back Down onto Road Image

### Display Lane Boundaries Plotted onto Road Image

onto Image

2\. I computed the inverse perspective transform

~~~python
Minv = cv2.getPerspectiveTransform(dst, src)
~~~

### Add Lane Curvature and Vehicle Position onto Road Image

to Image

## Pipeline (Video Processing)

**Rubric Criteria: Provide a link to your final video output.**

**Rubric Note: Your pipeline should perform well on entire project video (wobbly lines are ok, but no catastrophic failures that cause the car to drive off the road!)**

## Discussion

problems/issues faced

what hypothetical cases would cause your pipeline to fail


