import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# Gradient Thresholds and Color Spaces can be used to more easily
# identify lane markings on the road
# GradientThresholds:
# Has to do with capturing a property's magnitude and direction in an image. 
# Property in our case are the lane lines.
# We find the lane line magnitude and direction in an image. We can also focus 
# on other properties in an image.
class GradientThresholds:
    def __init__(self):
        self.img_m = None
        
    def apply_sobel_thresh(self, img, orient='x', ksize=(3,3), thresh=(0, 255)):
        """
            Calculate directional gradient.
            Identify pixels where the gradient of an image falls within a specified threshold range.
            Pass in image, choose orient to be x or y gradient, specify 
            threshold min and max range to select for binary_output. The 
            binary_output array is 1 where gradients are in threshold range,
            0 everywhere else.
        """
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Remove noise
        gray = cv2.GaussianBlur(gray, ksize, 0)
        
        # Take derivative in x or y 
        if orient == 'x':
            # Apply sobel for the x direction
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        elif orient == 'y':
            # Apply sobel for the y direction
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        # Take absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale the result to an 8-bit range (0-255)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Apply lower and upper thresholds to mask scaled gradient
        binary_image = np.zeros_like(scaled_sobel)
        # Apply 1's when scaled gradient is within threshold
        binary_image[(scaled_sobel >= thresh[0]) & 
                     (scaled_sobel <= thresh[1])] = 1
            
        # Return this mask as binary image
        return binary_image
    
    def apply_grad_mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        """
            Calculate gradient magnitude
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        # Calculate the gradient magnitude in both x and y direction
        grad_mag = np.sqrt( (sobelx**2) + (sobely**2) )
        # Scale to 8-bit and convert to type = np.uint8
        scale_factor = np.max(grad_mag)/255
        scaled_grad_mag = (grad_mag/scale_factor).astype(np.uint8)
        # Create a binary mask where mag thresholds are met
        binary_image = np.zeros_like(scaled_grad_mag)
        binary_image[(scaled_grad_mag >= mag_thresh[0]) & 
                     (scaled_grad_mag <= mag_thresh[1])] = 1

        # Return this mask as binary_image
        return binary_image
        
    def apply_grad_dir_thresh(self, img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
        """
            Calculate gradient direction
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take gradient in x and y
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        # absolute value of x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # Calculate the gradient direction
        dir_grad = np.arctan2(abs_sobely, abs_sobelx)      
        
        # Create a binary image where direction thresholds are met
        binary_image = np.zeros_like(dir_grad)
        binary_image[(dir_grad >= dir_thresh[0]) & (dir_grad <= dir_thresh[1])] = 1
        
        # Return binary_image
        return binary_image
        
    def apply_combined_thresh(self, combination_code, grad_x = None, grad_y = None, grad_mag = None, grad_dir = None):
        """
            Combine Gradient Thresholding binary images based on the gradients 
            already applied, they set private variables that can be used in this
            method. Choose based on code, which thresholds you'd combine:
            0: X-Sobel, Gradient Magnitude
            1: X-Sobel, Gradient Direction
            2: X-Sobel, Gradient Magnitude, Gradient Direction
            3: X-Sobel, Y-Sobel, Gradient Magnitude, Gradient Direction
        """
        combined = np.zeros_like(grad_x)
        if combination_code == 0:
            combined[ (grad_x == 1) & (grad_mag == 1) ] = 1
        elif combination_code == 1:
            combined[ (grad_x == 1) & (grad_dir == 1) ] = 1
        elif combination_code == 2: 
            combined[ (grad_x == 1) & (grad_mag == 1) & (grad_dir == 1) ] = 1  
        elif combination_code == 3:
            combined[ (grad_x == 1) & (grad_y == 1) & 
                      (grad_mag == 1) & (grad_dir == 1) ] = 1             
        else:
            print("Error: Choose a supported code for combined gradient")

        # Return binary result from multiple thresholds
        return combined

    def save_img(self, dst_path, filename, dst_img):
        """
        Save gradient thresholded image using OpenCV
        """
        # If filepath doesn't exist, create it
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        
        # Save binary image resulting from gradient thresholding
        plt.imsave(dst_path + filename, dst_img, cmap = "gray")
    
    def visualize(self, src_title, undist_img, dst_title, binary_img):
        """
        Visualize gradient thresholded image
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(undist_img, cmap = 'gray')
        ax1.set_title(src_title, fontsize=50)
        ax2.imshow(binary_img, cmap = 'gray')
        ax2.set_title(dst_title, fontsize=50)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.)