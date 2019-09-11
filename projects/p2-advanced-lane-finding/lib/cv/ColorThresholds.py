import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Color space is a specific organization of colors
# They provide a way to categorize colors and represent them in digital images

# RGB Thresholding doesn't work well under varying light conditions
# or under varying color like yellow

# HLS Thresholding isolates lightness (L), which varies most under
# different lighting conditions.
# H and S channels stay consistent in shadow or excessive brightness

# HLS can be used to detect lane lines of different colors under
    # different lighting conditions

# Hue - represents color independent of any change in brightness
# Lightness and Value - represent different ways to measure relative
    # lightness or darkness of a color
# Saturation - measurement of colorfulness
    # As colors get lighter (white), their saturation value is lower
    # Most intense colors (bright red, blue , yellow) have high saturation

class ColorThresholds:
    # Apply Grayscale Thresholding
    def apply_gray_thresh(self, img, thresh = (0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        binary_img = np.zeros_like(gray)
        binary_img[ (gray > thresh[0]) & (gray <= thresh[1]) ] = 1
        return binary_img
    
    # Thresholding individual RGB Color Channels
    def apply_r_thresh(self, img, thresh = (0, 255)):
        r_img = img[:,:,0]
        binary_img = np.zeros_like(r_img)
        binary_img[ (r_img >= thresh[0]) & (r_img <= thresh[1]) ] = 1
        return binary_img
    
    def apply_g_thresh(self, img, thresh = (0, 255)):
        g_img = img[:,:,1]
        binary_img = np.zeros_like(g_img)
        binary_img[ (g_img >= thresh[0]) & (g_img <= thresh[1]) ] = 1
        return binary_img        

    def apply_b_thresh(self, img, thresh = (0, 255)):
        b_img = img[:,:,2]
        binary_img = np.zeros_like(b_img)
        binary_img[ (b_img >= thresh[0]) & (b_img <= thresh[1]) ] = 1
        return binary_img  
    
    def apply_rgb_thresh(self, num_code, rgb_r = None, rgb_g = None, rgb_b = None):
        """
            Combine RGB Thresholding binary images based on the red, green and/or
            blue thresholds already applied, they set private variables that can be
            used in this method. Choose based on number code, which thresholds you'd
            combine:
            0: R Binary, G Binary
            1: R Binary, B binary
            2: G Binary, B Binary
            3: R Binary, G Binary, B Binary
        """
        combined = np.zeros_like(rgb_r)
        if num_code == 0:
            combined[ (rgb_r == 1) | (rgb_g == 1) ] = 1
        elif num_code == 1:
            combined[ (rgb_r == 1) & (rgb_b == 1) ] = 1
        elif num_code == 2: 
            combined[ (rgb_g == 1) & (rgb_b == 1) ] = 1  
        elif num_code == 3:
            combined[ ((rgb_r == 1) | (rgb_g == 1)) & (rgb_b == 1) ] = 1             
        else:
            print("Error: Choose a supported code for combined rgb")

        # Return binary result from multiple thresholds
        return combined
    
    # Thresholding individual HSL Color Channels
    def apply_h_thresh(self, img, thresh = (0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_img = hls[:,:,0]
        binary_img = np.zeros_like(h_img)
        binary_img[ (h_img >= thresh[0]) & (h_img <= thresh[1]) ] = 1
        return binary_img
    
    def apply_l_thresh(self, img, thresh = (0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_img = hls[:,:,1]
        binary_img = np.zeros_like(l_img)
        binary_img[ (l_img >= thresh[0]) & (l_img <= thresh[1]) ] = 1
        return binary_img
    
    def apply_s_thresh(self, img, thresh = (0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_img = hls[:,:,2]
        binary_img = np.zeros_like(s_img)
        binary_img[ (s_img >= thresh[0]) & (s_img <= thresh[1]) ] = 1
        return binary_img
    
    # Apply Combined HLS Thresholding
    def apply_hls_thresh(self, num_code, hls_h = None, hls_l = None, hls_s = None):
        """
            Combine HLS Thresholding binary images based on the hue, lightness
            and/or saturation thresholds already applied, they set private 
            variables that can be used in this method. Choose based on number 
            code, which thresholds you'd combine:
            # 0: H Binary, L Binary
            # 1: H Binary, S binary
            # 2: L Binary, S Binary
            # 3: H Binary, L Binary, S Binary  
        """
        combined = np.zeros_like(hls_h)
        if num_code == 0:
            combined[ (hls_h == 1) & (hls_l == 1) ] = 1
        elif num_code == 1:
            combined[ (hls_h == 1) | (hls_s == 1) ] = 1
        elif num_code == 2: 
            combined[ (hls_l == 1) & (hls_s == 1) ] = 1  
        elif num_code == 3:
            combined[ (hls_h == 1) | ((hls_s == 1) & (hls_l == 1)) ] = 1             
        else:
            print("Error: Choose a supported code for combined hls")

        # Return binary result from multiple thresholds
        return combined        
        
        
        h_binary = self.apply_h_thresh(img, thresh[0])
        l_binary = self.apply_l_thresh(img, thresh[1])
        s_binary = self.apply_s_thresh(img, thresh[2])
        combined = np.zeros_like(s_binary)
        combined[ (h_binary == 1) & (l_binary == 1) & (s_binary == 1) ] = 1
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
        Visualize color thresholded image
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(undist_img, cmap = 'gray')
        ax1.set_title(src_title, fontsize=50)
        ax2.imshow(binary_img, cmap = 'gray')
        ax2.set_title(dst_title, fontsize=50)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0.)