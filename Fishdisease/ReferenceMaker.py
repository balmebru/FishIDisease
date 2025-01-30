import cv2
import numpy as np
from ultralytics import YOLO
from Segmenter import Segmenter


class ReferenceMaker:
    def __init__(self, model_path):
        """"
        Initializing the model
        """
        self.model = YOLO(model_path)
        pass

    def detect_reference_object(self, image):
        """
        Detect and segment the reference object in an image
        
        """
        mask = Segmenter.segment_fish(image)  
        if mask is None:
            return None, None
        return mask

    def compute_correction_factors(self, image):
        """
        Computing brightness and colr correction factors
        """
        reference_mask = self.detect_reference_object(image)  
        ref_region = cv2.bitwise_and(image,image,mask= reference_mask)
        ## brightness
        avg_b = np.mean(ref_region)
        target_b = 128 # normalizing to mid
        b_factor = target_b / avg_b

        ## color

        mean_r, mean_g, mean_b = cv2.mean(ref_region)
        target_r, target_g, target_b = 128, 128, 128
        r_factor, g_factor, b_factor = target_r / mean_r, target_g / mean_g, target_b / mean_b

        return b_factor, (r_factor, g_factor, b_factor)
    

    def apply(self, image):

        """
        applying the computed factors to image
        """
        b_factor, c_factors = self.compute_correction_factors(self, image)
        r_factor, g_factor, b_factor = c_factors
        corrected = cv2.merge([
            np.clip(image[..., 0] * b_factor, 0, 255).astype(np.uint8),
            np.clip(image[..., 1] * g_factor, 0, 255).astype(np.uint8),
            np.clip(image[..., 2] * r_factor, 0, 255).astype(np.uint8)

        ])
        adjusted = np.clip(corrected * b_factor, 0, 255).astype(np.uint8)
        return adjusted
    
    def resize_to_reference(self, image, ref_bbox, target_size=(100, 100)):
        """
        Resizing the reference object to standard size.
        """
        x1, y1, x2, y2 = map(int, ref_bbox)
        ref_object = image[y1:y2, x1:x2]
        resized = cv2.resize(ref_object, target_size, interpolation=cv2.INTER_LINEAR)
        return resized

