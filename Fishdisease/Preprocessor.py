
import cv2

import numpy as np
import matplotlib.pyplot as plt
import pywt
from Segmenter import Segmenter
from ReferenceMaker import ReferenceMaker

class Preprocessor:
    def __init__(self):
        pass


    def apply_denoising(self, image):
        """
        Applies selective denoising using the bilateral filter
        """
        denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        return denoised_image

    def lighting_correction(self, image):
        corrected_image = ReferenceMaker.apply(image)
        return corrected_image
    

    def preprocessing(self,image):
        """
        Needs to correct the image based on a refrence object in the image
        Shoud detect the refrence object and correct the image based on the refrence object
        
        ideas: 
        Haar wavelet transform, which was then combined with histogram equalization

        imporved median filter
        Processing the whole original image with 5×5 standard median filter, a rough processed image will be got. 
        Subtracting pixel values of processed image from the original image respectively, 
        the mean value of the results will be taken as threshold to decide the pixels we interested in is a noisy pixel or not. 
        Specifically, if a pixel value of the subtraction result is greater than threshold,
            than the pixel will be classified as a noisy one and vice versa.


        """


        image_no_background = self.background_subtraction(image)
        denoised_image = self.apply_denoising(image_no_background)
        colour_corrected_image = self.lighting_correction(denoised_image)
        enhanced_image = self.haar_wavelet_histogram_equalization(colour_corrected_image)
        print("Image enhaced using haar wavelet")
        mask = self.get_noisy_pixel_mask(enhanced_image)
        print(" i NOW HAVE THE MASK FOR NOISY")
        print("The size of image is ", enhanced_image.shape[:2])
        print("the size of nosiy image is ", mask.shape)
        corrected_image = self.apply_median_filter_to_noisy_pixels(enhanced_image, mask)
        print("Done")
        normalized_image = self.normalize_image(enhanced_image)
        self.compare_image(image,normalized_image)
        return normalized_image

    def compare_image(self,image1,image2):
        """
        showes the images for comparison

        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image1)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(image2)
        ax[1].set_title("Corrected Image")
        ax[1].axis("off")
        plt.show()

        
        return

    def detect_fish(self,image):
        """
        Needs to detect and segment the fish in the image
        """
        mask = image
        return mask

    def characterize_fish(self,single_mask):
        """
        Needs to characterize a mask in the image
        """
        return


    def improved_median_filter():
        return

    def get_noisy_pixel_mask(self, image):
        """
        Identifies noisy pixels using a 5x5 median filter and thresholding.

        Parameters:
            image (np.array): Input RGB image as a NumPy array.

        Returns:
            np.array: Boolean mask identifying noisy pixels.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array.")

        processed_image = cv2.medianBlur(image, 5)
        difference = np.abs(image.astype(np.float32) - processed_image.astype(np.float32))
        threshold = np.mean(difference)

        # Create noisy mask
        noisy_mask = (difference > threshold).astype(np.uint8)

        print("Shape of difference:", difference.shape)  
        print("Shape of noisy_mask before squeeze:", noisy_mask.shape)  

        # Ensure (H, W) shape by taking max across channels
        if noisy_mask.ndim == 3:  
            noisy_mask = np.max(noisy_mask, axis=-1)  
        print("Shape of noisy_mask after squeeze:", noisy_mask.shape)  

        return noisy_mask  




    def apply_median_filter_to_noisy_pixels(self,image, noisy_mask):
        """
        Applies a 5x5 median filter only to the noisy pixels identified by the mask.

        Parameters:
            image (np.array): Original RGB image as a NumPy array.
            noisy_mask (np.array): Boolean mask identifying noisy pixels.

        Returns:
            np.array: Filtered RGB image with median filter applied only to noisy pixels.
        """
        # Ensure inputs are NumPy arrays
        if not isinstance(image, np.ndarray) or not isinstance(noisy_mask, np.ndarray):
            raise ValueError("Both image and noisy_mask must be NumPy arrays.")
        
        # Check if shapes match
        if image.shape[0:] != noisy_mask.shape:
            raise ValueError("The dimensions of the image and mask must match.")

        # Create a copy of the original image to modify
        filtered_image = image.copy()

        # Apply median filter to the entire image
        median_filtered = cv2.medianBlur(image, 5)

        # Replace only the noisy pixels with their median-filtered values
        filtered_image[noisy_mask.astype(bool)] = median_filtered[noisy_mask.astype(bool)]

        return filtered_image

    def haar_wavelet_histogram_equalization(self, image):
        """
        Apply Haar wavelet transform and histogram equalization to an image.
        
        Parameters:
        image (numpy.ndarray): Input image (grayscale)
        
        Returns:
        numpy.ndarray: Processed image
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Haar Wavelet Transform
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        # Step 2: Histogram Equalization on approximation coefficients
        cA_equalized = cv2.equalizeHist(np.uint8(cA))
        
        # Optional: Apply histogram equalization to detail coefficients
        # Be cautious as this can amplify noise
        cH_equalized = cv2.equalizeHist(np.uint8(np.abs(cH)))
        cV_equalized = cv2.equalizeHist(np.uint8(np.abs(cV)))
        cD_equalized = cv2.equalizeHist(np.uint8(np.abs(cD)))
        
        # Reconstruct the image
        reconstructed = pywt.idwt2((cA_equalized, (cH_equalized, cV_equalized, cD_equalized)), 'haar')
        
        # Normalize the output to 0-255 range
        reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
        
        return np.uint8(reconstructed)
    
    def edge_enhancement(self, image):
        """
        Laplacian filter
        """

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        enhanced_image = cv2.convertScaleAbs(laplacian)
        enhanced_image = cv2.addWeighted(image, 1.5, enhanced_image, -0.5, 0)
        return enhanced_image
    
    def normalize_image(self, image):
        """
        normalize the image pixel values to the range 0 to 1 
        """

        return cv2.normalize(image, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)