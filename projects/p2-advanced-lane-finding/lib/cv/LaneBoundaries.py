import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class LaneBoundaries:
    def __init__(self):
        # Binary Warped Image
        self.binary_warped_m = None
        
        # Original image right after distortion correction applied
        self.undist_img_m = None
        
        # Attributes related to fit the lines with a polynomial
        self.ploty_m = None
        self.left_fitx_m = None
        self.right_fitx_m = None
        
        # Inverse Perspective Matrix for warping the blank img back to original img
        self.Minv_m = None
        
        # Overlayed undistorted image with lane boundaries detected
        self.result_m = None
        
    def set_warped_binary_img(self, binary_warped):
        self.binary_warped_m = binary_warped
        
    def set_original_undist_img(self, origin_undist_img):
        """
            Sets private original image with original image right after distortion
            correction was applied
        """
        self.undist_img_m = origin_undist_img
        
    def set_fit_lines_poly(self, ploty, left_fit, right_fit):
        self.ploty_m = ploty
        
        # Generate x and y values for plotting
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]  
        
        self.left_fitx_m = left_fitx
        self.right_fitx_m = right_fitx
        
    def set_minv(self, Minv):
        self.Minv_m = Minv
        
    def set_lane_curvature_radius(self, left_curverad, right_curverad, units):
        """
            Sets left and right curvature radius and units, so it can be displayed
            on lane boundaries image
        """
        self.left_curverad_m = left_curverad
        self.right_curverad_m = right_curverad
        self.curverad_units_m = units
        
    def set_lane_curvature_angle(self, l_angle_curve, r_angle_curve, units):
        """
            Sets left and right curvature radius and units, so it can be displayed
            on lane boundaries image
        """
        self.l_angle_curve_m = l_angle_curve
        self.r_angle_curve_m = r_angle_curve
        self.angle_units_m = units        
        
    def set_vehicle_position(self, dist_center, position_units, side_center):
        self.dist_center_m = dist_center
        self.units_m = position_units
        self.side_center_m = side_center

    def set_img_text_properties(self, font_family=cv2.FONT_HERSHEY_SIMPLEX, font_color=(255,255,255), font_size=2, font_thickness = 2, line_type = cv2.LINE_AA):
        """
            Set image text properties to configure how text is displayed on an 
            image, which affects overlay_lane_curvature() and
            overlay_vehicle_position()
        """
        self.font_family_m = font_family
        self.font_color_m = font_color
        self.font_size_m = font_size
        self.font_thickness_m = font_thickness
        self.line_type_m = line_type        
        
    def set_overlayed_lane_boundary(self, lane_boundary_img):
        """
            Sets image that already has the lane boundary overlayed onto it
        """
        self.result_m = lane_boundary_img
        
    def overlay_lane_boundaries(self):
        """
            Warps the detected lane boundaries onto the original image
            binary_warped, left_fitx, right_fitx, ploty
        """
        # Establish original image size as tuple
        undist_img_size = (self.undist_img_m.shape[1], self.undist_img_m.shape[0])
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped_m).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx_m, self.ploty_m]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx_m, self.ploty_m])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Warp blank back to original image space, inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv_m, undist_img_size)
        
        # Combine the result with the original image for lane boundaries to appear
        self.result_m = cv2.addWeighted(self.undist_img_m, 1, newwarp, 0.3, 0)   
        
    def overlay_radius_curvature(self):
        """
            Adds Lane Curvature Radius text onto the overlayed lane boundaries image
        """
        img_width = self.undist_img_m.shape[1]
        img_heiht = self.undist_img_m.shape[0]
        # Calculate lane curvature average
        lane_curverad_avg = (self.left_curverad_m + self.right_curverad_m)/2
        # Set lane curvature radius average in string
        lane_curverad_str = "Lane Curvature Radius = {:.0f} {}".format(lane_curverad_avg, self.curverad_units_m)
        
        # Put lane curvature string onto image result
        cv2.putText(self.result_m, lane_curverad_str, (50, 50), self.font_family_m, self.font_size_m, self.font_color_m, self.font_thickness_m, self.line_type_m)
        
    def overlay_angle_curvature(self):
        """
            Adds Lane Curvature Radius text onto the overlayed lane boundaries image
        """
        img_width = self.undist_img_m.shape[1]
        img_heiht = self.undist_img_m.shape[0]
        # Calculate lane curvature average
        lane_curve_angle_avg = (self.l_angle_curve_m + self.r_angle_curve_m)/2
        # Set lane curvature radius average in string
        lane_curve_angle_str = "Lane Curvature Angle = {:.0f} {}".format(lane_curve_angle_avg, self.angle_units_m)
        
        # Put lane curvature string onto image result
        cv2.putText(self.result_m, lane_curve_angle_str, (50, 150), self.font_family_m, self.font_size_m, self.font_color_m, self.font_thickness_m, self.line_type_m)
        
    def overlay_vehicle_position(self):
        """
            Adds text for Vehicle Position with respect to center of the lane
            onto the overlayed lane boundaries and lane curvature image
        """
        # Set vehicle position string
        vehicle_position_str = "Vehicle is {:.2f} {} {}".format(self.dist_center_m, self.units_m, self.side_center_m)
        
        # Put vehicle position string onto image result
        cv2.putText(self.result_m, vehicle_position_str, (50, 250), self.font_family_m, self.font_size_m, self.font_color_m, self.font_thickness_m, self.line_type_m)
        
    def get_overlayed_image(self):
        """
            Returns overlayed image that now has lane boundaries, lane curvature 
            and vehicle position.
        """
        return self.result_m
        
    def visualize(self):
        """
            Visualizes the detected lane boundary overlayed onto the 
            undistorted image
        """
        plt.figure(figsize = (15, 15))
        plt.imshow(self.result_m)
        
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