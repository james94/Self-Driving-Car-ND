# Quiz Solution: Hough Transform

For quadrilateral region selection, the following parameters were defined:

~~~python
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
~~~

For Hough Space Grid, `rho` is 2 pixels, `theta` is 1 degree (pi/180 radians), `threshold` is 15, which means at least 15 points in image space n need to be associated with each line segment. `min_line_length` is 40 pixels and `max_line_gap` is 20 pixels.

These parameters allowed for picking up the lane lines and nothing else, one possible solution.

![exit-ramp.jpg](../../images/exit-ramp.jpg)

**Figure 1: Original Image**

![exit_ramp_hough_masked.jpg](../../images/exit_ramp_hough_masked.jpg)

**Figure 2: Edge Detection, Region Masking and Hough Transform applied**

