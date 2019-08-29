# Quiz Solution: Canny to Detect Lane Lines

The `kernel_size` was set `5` for Gaussian smoothing. The `low_threshold` was set to `50` and `high_threshold` was set to `150` for Canny. These selections extract the lane lines while minimizing the edges detected in the rest of the image. The result can be seen in figure 2.

![exit-ramp.jpg](../../images/exit-ramp.jpg)

**Figure 1: The original image**

![edges-exit-ramp-v2.jpg](../../images/edges-exit-ramp-v2.jpg)

**Figure 2: Canny Edge Detection Applied**
