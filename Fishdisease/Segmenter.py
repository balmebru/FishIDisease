
import os
from ultralytics.data.annotator import auto_annotate
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from typing import List, Dict, Tuple

class Segmenter:
    def __init__(self):
        pass

    def detect_fish(self, image):
        pass

    def segment_fish(self, image):
        pass

    def extract_mask(self, segmented_image):
        pass



    def autoannotate_fish(self,data,detection_model,sam_model,output_dir_fish):

        """
        Takes a input directory with images and runs the autoannotate function on the images
        
        
        """
        auto_annotate(data=data,det_model=detection_model,sam_model=sam_model,output_dir=output_dir_fish,conf=0.35,iou=0.9,max_det=1)

        return
    
    

    def segment_pictures_with_input_prompt(self, sam_model_path: str, image_path: str, input_prompt: str):
        """
        This function uses the SAM model to segment objects in images based on an input prompt.
        
        Args:
            sam_model_path (str): Path to the SAM model checkpoint.
            image_path (str): Directory path containing images.
            input_prompt (str): User prompt for segmentation guidance.
        """
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_model_path)
        predictor = SamPredictor(sam)

        for image_file in os.listdir(image_path):
            image_full_path = os.path.join(image_path, image_file)

            # Read the image
            image = cv2.imread(image_full_path)
            if image is None:
                print(f"Failed to read image: {image_full_path}")
                continue

            predictor.set_image(image)
            height, width, _ = image.shape

            # Define prompt coordinates relative to image dimensions (here in the middle)
            input_point = np.array([[0.5 * width, 0.5 * height]])
            input_label = np.array([1])

            # Perform segmentation
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            # Extract contours from the mask
            mask = masks[0]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Save contours to YOLOv8 format
            output_file_path = os.path.splitext(image_full_path)[0] + ".txt"
            self.save_sam_to_yolov8_format(contours, (height, width), output_file_path)

    def save_sam_to_yolov8_format(self, contours: List[np.ndarray], image_shape: Tuple[int, int], output_path: str, class_id: int = 1) -> None:
        """
        Save bounding boxes derived from contours in YOLO format.

        Args:
            contours (List[np.ndarray]): List of contours.
            image_shape (Tuple[int, int]): Shape of the input image (height, width).
            output_path (str): Path to save the YOLO format file.
            class_id (int): Class ID for YOLO format.
        """
        height, width = image_shape
        
        with open(output_path, "w") as file:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Normalize coordinates
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                width_normalized = w / width
                height_normalized = h / height

                # Write to file
                file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f}\n")



        return