
import cv2

import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def background_subtraction(self, image):
        pass

    def apply_denoising(self, image):
        pass

    def lighting_correction(self, image, correction_factors):
        pass
    

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



        mask = self.get_noisy_pixel_mask(image)
        corrected_image = self.apply_median_filter_to_noisy_pixels(image,mask)
        self.compare_image(image,corrected_image)
        return corrected_image

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


    def haar_wavelet_transform_and_histogramm_equalisation():
        return

    def improved_median_filter():
        return

    def get_noisy_pixel_mask(self,image):
        """
        Removes noisy pixels from an RGB image using a 5x5 median filter and thresholding.

        Parameters:
            image (np.array): Input RGB image as a NumPy array.

        Returns:
            np.array: Filtered RGB image.
        """
        # Ensure the input is a NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array representing an RGB image.")
        
        # Apply a 5x5 median filter to the original image
        processed_image = cv2.medianBlur(image, 5)
        
        # Compute the difference between the original and processed images
        difference = np.abs(image.astype(np.float32) - processed_image.astype(np.float32))
        
        # Calculate the mean value of the difference as the threshold
        threshold = np.mean(difference)
        
        # Create a mask to identify noisy pixels
        noisy_mask = np.any(difference > threshold, axis=-1)  # Check if any channel exceeds the threshold
        
        
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
        if image.shape[:2] != noisy_mask.shape:
            raise ValueError("The dimensions of the image and mask must match.")

        # Create a copy of the original image to modify
        filtered_image = image.copy()

        # Apply median filter to the entire image
        median_filtered = cv2.medianBlur(image, 5)

        # Replace only the noisy pixels with their median-filtered values
        filtered_image[noisy_mask] = median_filtered[noisy_mask]

        return filtered_image

    def haar_wavelet_histogram_equalization(image):
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