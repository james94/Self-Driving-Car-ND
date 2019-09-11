import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import cv2
import os

# Mathematically we can characterize perspective by saying that in real world 
# coordinates (X, Y, Z), the greater the magnitude of an object's Z coordinate or
# distance from the camera, the smaller it will appear in a 2D image

# The Perspective Transform uses this information to transform an image, it transforms
# the apparent Z coordinate of the object points, which in turn changes that object's
# 2D representation

# A Perspective Transform warps the image and drags points towards or pushes them
# away from the camera to change the apparent perspective

# CameraPerspective applies a perspective transform on camera images to warp it into
# different perspectives: Bird's Eye View, etc
class CameraPerspective:
    def birds_eye_view(self, img):
        """
            Apply Bird's Eye View Transform to Camera Image for a Top-Down View
        """
        imshape = img.shape
        
        img_size = (img.shape[1], img.shape[0])
        
        # source points
        src = np.array( 
            [[imshape[1]*0.145, imshape[0]], # bottom left
             [imshape[1]*0.462, imshape[0]*0.62], # top left
             [imshape[1]*0.535, imshape[0]*0.62], # top right
             [imshape[1]*0.883, imshape[0]]], # bottom right
            dtype = np.float32 )      
        
        # destination points
        dst = np.array(
            [[imshape[1]*0.24, imshape[0]], # bottom left
             [imshape[1]*0.24, imshape[0]*0], # top left
             [imshape[1]*0.75, imshape[0]*0], # top right
             [imshape[1]*0.75, imshape[0]]], # bottom right
            dtype = np.float32)
        
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Could compute the inverse also by swapping the input parameters
        self.Minv_m = cv2.getPerspectiveTransform(dst, src)
        
        # Create warped image - uses linear interpolation
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        
        return warped

    def get_minv(self):
        """
            Returns Minv, the inverse perspective transform
        """
        return self.Minv_m
    
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
    
    def visualize(self, src_title, undist_img, dst_title, binary_img):
        """
        Visualize color thresholded image
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(undist_img, cmap = 'gray')
        ax1.set_title(src_title, fontsize=50)
        ax2.imshow(binary_img, cmap = 'gray')
        ax2.set_title(dst_title, fontsize=50)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.)      