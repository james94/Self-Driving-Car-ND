import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Prerequisite: Have applied camera calibration, thresholding and perspective
    # transform to a road image, results in a binary warped image where lane lines
    # stand out
    
# LaneLineDetection takes in a binary image and uses Histogram Peaks, 
# Sliding Window Search and Search from Prior to decide which pixels 
# are part of the lines, which belong to left line and which belong to right line. 
# LaneLineDetection then fits a polynomial to each lane line and gives the user
# the option to display it

class LaneLineDetection:
    def __init__(self):
        """
            LaneLineDetection uses the sliding window method to find all our pixels
            So, the constructor initially sets the hyperparameters for the sliding
            window, but these values can be customized before fitting the 
            polynomial to the line by first calling 'setup_sw_hyperparameters()' 
            method
        """
        # Sliding Window Base Hyperparameters
        # Number of Sliding Windows
        self.nwindows_m = 9
        # Width of the windows +/- margin
        self.margin_m = 100
        # Minimum number of pixels found to recenter window
        self.minpix_m = 50      
        
        # Camera image has 720 relevant pixels or 30 meters long in the y-dimension
        self.ym_per_pix_m = 30/720 # Meters per Pixel in y dimension
        
        # Camera image has 700 relevant pixels or 3.7 meters wide in the x-dimension
        # 200 pixels were used on the left and 900 on the right
        self.xm_per_pix_m = 3.7/700 # Meters per Pixel in x dimension        
        
    # Histogram Peaks
        
    def histogram_peaks(self, binary_warped):
        """
            With this histogram, the pixel values are added up along each column 
            in the lower half of the binary image. Pixels are either 0 or 1, so 
            the two most prominent peaks are a good indicator of the x-pos of the
            base of the lane lines. With this info, we can determine where the 
            lane lines are.
        """
        # Grabs only the bottom half of the image
        # Why? Lane lines are likely to be vertical nearest to the car
        bottom_half = binary_warped[binary_warped.shape[0]//2:,:]
        
        # Histogram is the sum of pixel values along each column in the image
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis = 0)
        
        return histogram
    
    def visualize_hist(self, dst_title, histogram):
        """
            Visualize resulting historgram from histogram_peaks() method
        """
        plt.plot(histogram)
        plt.title(dst_title, {'fontsize': 20})
#         plt.show()
        
    def visualize_lanes_and_hist(self, src_title, undist_img, graph_title, graph):
        """
        Visualize warped bird's eye view image next to histogram peaks graph
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(undist_img, cmap = 'gray')
        ax1.set_title(src_title, fontsize=50)
        ax2.plot(graph)
        ax2.set_title(graph_title, fontsize=50)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.)     
#         plt.show()   
        
    # Sliding Window    
        
    def split_histogram(self, histogram):
        """
            Split the histogram into two sides, one for each lane line
        """
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        self.midpoint_m = np.int(histogram.shape[0]//2)
        self.leftx_base_m = np.argmax(histogram[:self.midpoint_m])
        self.rightx_base_m = np.argmax(histogram[self.midpoint_m:]) + self.midpoint_m
        
    def get_xint_polynomials(self):
        """
            Retrieves x-intercepts from left and right polynomials
        """
        return self.leftx_base_m, self.rightx_base_m
    
    def setup_sw_hyperparameters(self, nwindows, margin, minpix):
        """
            Optional: Set a few hyperparamters for our sliding windows, so they are
            set up to iterate across the binary activations in the image. 
        """
        # Hyperparameters
        # Choose the number of sliding windows
        self.nwindows_m = nwindows
        # Set the width of the windows +/- margin
        self.margin_m = margin
        # Set minimum number of pixels found to recenter window
        self.minpix_m = minpix
    
    def setup_sw(self, binary_warped):
        """
            Set up sliding windows
        """
        # Set height of windows - based on nwindows above and image shape
        self.window_height_m = np.int(binary_warped.shape[0]//self.nwindows_m)
        # Identify x and y positions of all nonzero (i.e. activated) pixels in image
        nonzero = binary_warped.nonzero()
        self.nonzeroy_m = np.array(nonzero[0])
        self.nonzerox_m = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        self.leftx_current_m = self.leftx_base_m
        self.rightx_current_m = self.rightx_base_m
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds_m = []
        self.right_lane_inds_m = []
    
    def track_curvature(self, binary_warped):
        """
            Prerequisite: Set what the windows look like and have a starting point
            Loop through nwindows to track curvature. The given window slides
            left or right if it finds the mean position of activated pixels within
            the window have shifted.
        """
        # Create a class member output image to draw on and visualize result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Step through the windows one by one
        for window in range(self.nwindows_m):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*self.window_height_m
            win_y_high = binary_warped.shape[0] - window*self.window_height_m
            # Find the four below boundaries of the window
            win_xleft_low = self.leftx_current_m - self.margin_m
            win_xleft_high = self.leftx_current_m + self.margin_m
            win_xright_low = self.rightx_current_m - self.margin_m
            win_xright_high = self.rightx_current_m + self.margin_m
            
            # Draw the window boundaries on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            # Identifies the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy_m >= win_y_low) & 
                              (self.nonzeroy_m < win_y_high) & 
                              (self.nonzerox_m >= win_xleft_low) & 
                              (self.nonzerox_m < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((self.nonzeroy_m >= win_y_low) &
                               (self.nonzeroy_m < win_y_high) &
                               (self.nonzerox_m >= win_xright_low) &
                               (self.nonzerox_m < win_xright_high)).nonzero()[0]
    
            # Append these indices to the lists
            self.left_lane_inds_m.append(good_left_inds)
            self.right_lane_inds_m.append(good_right_inds)
            
            # If the number of pixels found > minpix pixels, recenter next window
            # `rightx_current` or `leftx_current` on their mean position
            if len(good_left_inds) > self.minpix_m:
                self.leftx_current_m = np.int(np.mean(self.nonzerox_m[good_left_inds]))
            if len(good_right_inds) > self.minpix_m:
                self.rightx_current_m = np.int(np.mean(self.nonzerox_m[good_right_inds]))
         
        # Concatenate arrays of indices (previously was a list of lists of pixels)
        try:
            self.left_lane_inds_m = np.concatenate(self.left_lane_inds_m)
            self.right_lane_inds_m = np.concatenate(self.right_lane_inds_m)
        except ValueError:
            # Avoids error if the above isn't implemented fully
            pass
        
        # Extract left and right line pixel positions
        self.leftx_m = self.nonzerox_m[self.left_lane_inds_m]
        self.lefty_m = self.nonzeroy_m[self.left_lane_inds_m]
        self.rightx_m = self.nonzerox_m[self.right_lane_inds_m]
        self.righty_m = self.nonzeroy_m[self.right_lane_inds_m]
        
        return out_img

    def find_lane_pixels(self, binary_warped, histogram):
        """
            Uses Histogram peaks and Sliding Window method to find all pixels
            belonging to each line (left and right line)
        """
        self.split_histogram(histogram)
        self.setup_sw(binary_warped)
        return self.track_curvature(binary_warped)
    
    def fit_polynomial(self, binary_warped):
        """
            Fits a polynomial to each lane line
        """
        # Find our lane pixels
        # Fit a second order polynomial to each line using `np.polyfit`
        self.left_fit_m = np.polyfit(self.lefty_m, self.leftx_m, 2)
        self.right_fit_m = np.polyfit(self.righty_m, self.rightx_m, 2)
        
        # Generate x and y values for plotting
        self.ploty_m = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = self.left_fit_m[0]*self.ploty_m**2 + self.left_fit_m[1]*self.ploty_m + self.left_fit_m[2]
            right_fitx = self.right_fit_m[0]*self.ploty_m**2 + self.right_fit_m[1]*self.ploty_m + self.right_fit_m[2]
        except TypeError:
            # Avoids an error if `left_fit` and `right_fit` 
            # are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*self.ploty_m**2 + 1*self.ploty_m
            right_fitx = 1*self.ploty_m**2 + 1*self.ploty_m
            
        return left_fitx, right_fitx
    
    def get_fit_polynomial_data(self):
        """
            Returns second order polynomial to each line
        """
        return self.ploty_m, self.left_fit_m, self.right_fit_m
        
    def visualize_fit_polynomial(self, out_img, left_fitx, right_fitx, dst_title):
        """
            Visualize the sliding windows per line and the fit polynomial per
            lane line
        """
        # Color left lane line red on out_img
        out_img[self.lefty_m, self.leftx_m] = [255, 0, 0]
        # Color right lane line blue on out_img
        out_img[self.righty_m, self.rightx_m] = [0, 0, 255]
        
        # Plots the left polynomial on the lane line
        plt.plot(left_fitx, self.ploty_m, color = 'yellow')
        # Plots the right polynomial on the lane line
        plt.plot(right_fitx, self.ploty_m, color = 'yellow')
        # Set title for figure with sliding windows and fit polynomials
        plt.title(dst_title, {'fontsize': 20})
        # Display out_img with sliding windows plotted per lane line as green,
        # left lane line red, right lane line blue and each polynomial
        # plotted per lane line as yellow
        plt.imshow(out_img)
        
    def search_around_poly(self, binary_warped):
        """
            Uses polynomial left_fit and right_fit values from the previous frame
        """
        # Set the area of search based on activated x-values within the
        # +/- margin of our polynomial function
        self.left_lane_inds_m = ((self.nonzerox_m > 
                           (self.left_fit_m[0]*(self.nonzeroy_m**2) +
                            self.left_fit_m[1]*self.nonzeroy_m +
                            self.left_fit_m[2] - self.margin_m)) & 
                          (self.nonzerox_m < 
                           (self.left_fit_m[0]*(self.nonzeroy_m**2) +
                            self.left_fit_m[1]*self.nonzeroy_m +
                            self.left_fit_m[2] + self.margin_m)))
        
        self.right_lane_inds_m = ((self.nonzerox_m >
                            (self.right_fit_m[0]*(self.nonzeroy_m**2) +
                             self.right_fit_m[1]*self.nonzeroy_m +
                             self.right_fit_m[2] - self.margin_m)) &
                           (self.nonzerox_m < 
                            (self.right_fit_m[0]*(self.nonzeroy_m**2) +
                             self.right_fit_m[1]*self.nonzeroy_m +
                             self.right_fit_m[2] + self.margin_m)))
        
        # Again, extract left and right line pixel positions
        self.leftx_m = self.nonzerox_m[self.left_lane_inds_m]
        self.lefty_m = self.nonzeroy_m[self.left_lane_inds_m]
        self.rightx_m = self.nonzerox_m[self.right_lane_inds_m]
        self.righty_m = self.nonzeroy_m[self.right_lane_inds_m]
        
        # Fit new polynomials
        left_fitx, right_fitx = self.fit_polynomial(binary_warped)
        return left_fitx, right_fitx
        
    def visualize_sap(self, binary_warped, left_fitx, right_fitx):
        """
            Visualize the area around each line in green and the fit polynomial per
            lane line in yellow. The left line is red and right line is blue.
        """
        # Create an image to draw on and an image to show selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy_m[self.left_lane_inds_m], self.nonzerox_m[self.left_lane_inds_m]] = [255, 0, 0]
        out_img[self.nonzeroy_m[self.right_lane_inds_m], self.nonzerox_m[self.right_lane_inds_m]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x andy points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin_m, self.ploty_m]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin_m, self.ploty_m])))])
        # left line pts for left_line_window1 and left_line_window2
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
                            
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin_m, self.ploty_m]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin_m, self.ploty_m])))])
        # right line pts for right_line_window1 and right_line_window2                    
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
                            
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, self.ploty_m, color='yellow')
        plt.plot(right_fitx, self.ploty_m, color='yellow')
        # View Visualization of Search around Polynomial 
        plt.imshow(result)
        
    def save_img(self, dst_path, filename, dst_img):
        """
            Save image using OpenCV during bird's eye view transformation process,
            such as warped image
        """
        # If filepath doesn't exist, create it
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        
        # Save binary image resulting from gradient thresholding
        plt.imsave(dst_path + filename, dst_img, cmap = "gray")
    
    def save_fig(self, dst_path, filename):
        """
            Save figure using OpenCV during bird's eye view transformation process,
            such as source_points, destination_points, etc
        """
        # If filepath doesn't exist, create it
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        
        # Save current figure
        plt.savefig(dst_path + filename)