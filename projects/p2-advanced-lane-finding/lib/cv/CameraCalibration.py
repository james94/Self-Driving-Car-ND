import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import glob
import cv2
import os
# CameraCalibration class removes inherent distortions from the camera that can affect its perception of the world
class CameraCalibration:
    def __init__(self, nx, ny, cam_cal_dfp):
        # nx = corners for a row
        self.m_nx = nx
        # ny = corners for a column
        self.m_ny = ny
        self.m_cal_dfp = cam_cal_dfp
        # Distorted Image
        self.dist_img_m = None
        # list of calibration images
        self.m_images = glob.glob(self.m_cal_dfp)
        self.m_objp = self.get_prepared_objp()
        # Arrays to store object points and image points from all the images
        self.m_objpoints, self.m_imgpoints = self.extract_obj_img_points()
    
    def get_prepared_objp(self):
        """
        Prepare object points, like (0,0,0), (1,0,0) ..., (6,5,0)
        """
        objp = np.zeros( (self.m_ny * self.m_nx, 3), np.float32 )
        objp[:,:2] = np.mgrid[0:self.m_nx, 0:self.m_ny].T.reshape(-1,2)
        return objp
        
    def extract_obj_img_points(self):
        """
        Extract 3D Object Points and 2D Image Points
        """
        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane
        # Step through calibration image list and find chessboard corners
        for counter_x, fname in enumerate(self.m_images):
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.m_nx,self.m_ny), None)
            # As long as ret is True, then corners found
            if ret == True:
                # Extract object 3D and image 2D points
                objpoints.append(self.m_objp)
                imgpoints.append(corners)
                # Draw and display corners
                # cv2.drawChessboardCorners(img, (self.m_nx,self.m_ny), corners, ret)
                # write_name = "corners_found"+str(counter_x)+".jpg"
                # cv2.imwrite(write_name, img)
        return objpoints, imgpoints
    
    def cmpt_mtx_and_dist_coeffs(self, src_img_fpath):
        """
        Compute Camera Calibration Matrix and Distortion Coefficients using a set of
        chessboard images (initial), then curved lane line images.
        
        Returns distorted image, camera calibration matrix and distortion coefficients
        """
        # Test undistortion on a distorted image
        dist_img = mpimg.imread(src_img_fpath)
        # Get image size, which will be needed for calibrateCamera()
        img_size = (dist_img.shape[1], dist_img.shape[0])
        # Do camera calibration given object 3D points and image 2D points
        ret, mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(self.m_objpoints,
                                              self.m_imgpoints, img_size, None, None)    
        return mtx, dist_coeff
    
    def set_dist_img(self, src_img_fpath):
        """
            Sets private distorted image by reading in image with mpimg.imread()
        """
        # Read in RGB distorted image to self.dist_img_m
        self.dist_img_m = mpimg.imread(src_img_fpath)
        
        #self.dist_img_m = cv2.imread(src_img_fpath, cv2.IMREAD_COLOR)
        # Apply Trasnparent API for hardware acceleration when read src path img
        #self.dist_img_m = cv2.UMat(cv2.imread(src_img_fpath, cv2.IMREAD_COLOR))
    
    def rescale_img(self, width, height):
        """
            Downscale image resolution
        """
        # Checks if user provided dimensions greater than image dimension
        if(width > self.dist_img_m.shape[1] and height > self.dist_img_m.shape[0]):
            # Set dimension back to original image dimension
            width = self.dist_img_m.shape[1]
            height = self.dist_img_m.shape[0]
        # Use smaller dimension
        dim = (width, height)
        self.dist_img_m = cv2.resize(
            self.dist_img_m, dim, interpolation=cv2.INTER_AREA
        )
        
    def correct_distortion(self, mtx, dist_coeff, dist_img = None):
        """
        Apply Distortion Correction on an image by passing computed camera calibration
        matrix and distortion coefficients into undistort().
        
        Returns distorted image and undistorted image
        """
        # Using camera calibration matrix and distortion coefficients to undistort img
        if dist_img is not None:
            self.dist_img_m = dist_img
#         elif(self.dist_img_m == None):
#             print("Error: self.dist_img_m = None")
        
        undist_img = cv2.undistort(self.dist_img_m, mtx, dist_coeff, None, mtx)
        # Retrieve distorted image and undistorted image
        return self.dist_img_m, undist_img
    
    def save_img(self, dst_path, filename, dst_img, mtx, dist_coeff):
        """
        Save undistorted image using OpenCV and then pickle
        """
        # If filepath doesn't exist, create it
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)        
        
        # Convert dst_img from RGB to BGR, so image is saved wout color space issues 
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        # Save tested image after corrected distortion
        cv2.imwrite(dst_path + filename, dst_img)
        # Save camera calibration result for later use
        dist_pickle = {}
        # Camera Matrix is used to perform transformation from distorted to undistorted
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist_coeff
        filename = Path(filename)
        filename_wo_ext = str(filename.with_suffix(''))
        pickle.dump( dist_pickle, open(dst_path + filename_wo_ext + "_pickle.p", "wb") )    
    
    def visualize(self, src_title, src_img, dst_title, dst_img):
        """
        Visualize original distorted image and undistorted image using Matplotlib
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(src_img)
        ax1.set_title("Original: " + src_title, fontsize=30)
        ax2.imshow(dst_img)
        ax2.set_title("Undistorted: " + dst_title, fontsize=30)