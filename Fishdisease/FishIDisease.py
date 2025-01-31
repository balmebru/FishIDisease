import os
import cv2
import numpy as np
from .Segmenter import Segmenter

class FishIDisease:
    def __init__(self):
        pass

    def load_video(self, path: str):
        pass

    def run_pipeline(self):
        pass

    def export_results(self, output_path: str):
        pass

    def autoannotate_fish_directory(self, data, detection_model, sam_model, output_dir_fish):
        

        """
        Autoannotates the fishes in the given directory. And outputs the annotated images in the output directory.
        """
        segmenter_instance = Segmenter()
        segmenter_instance.autoannotate_fish(data=data,detection_model=detection_model,sam_model=sam_model,output_dir_fish=output_dir_fish)
        print("Done")


    def view_segmentation_after_autoanno(self,image_dir=None, segmentation_dir=None):
        """
        Visualizes segmentation contours from YOLOv11 output format.
        Saves the resulting images with visible contours to a new directory.
        """
        # Locate directories
        seg_dir = segmentation_dir
        output_dir = os.path.join(image_dir, "segmentation_results")
        os.makedirs(output_dir, exist_ok=True)

        # Process each image with a corresponding mask if present
        seg_files = {os.path.splitext(os.path.basename(seg_name))[0]: seg_name for seg_name in os.listdir(seg_dir)}

        for img_name in os.listdir(image_dir):
            img_base_name = os.path.splitext(os.path.basename(img_name))[0]
            if img_base_name in seg_files:
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(seg_dir, seg_files[img_base_name])

                # Load the original image
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    continue

                height, width = orig_img.shape[:2]

                # Load contour points from the YOLO format file
                with open(mask_path, 'r') as file:
                    contours = []
                    for line in file:
                        # Split the line into components
                        components = line.strip().split()
                        
                        # Skip the first element (id) and then pair x and y coordinates
                        points = [
                            (int(float(components[i]) * width), int(float(components[i+1]) * height))
                            for i in range(1, len(components), 2)
                        ]
                        
                        # Append the contour points as a numpy array
                        contours.append(np.array(points, dtype=np.int32))

                # Draw contours over the original image
                img_with_contours = orig_img.copy()
                cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

                # Save the resulting image
                output_path = os.path.join(output_dir, img_name)
                cv2.imwrite(output_path, img_with_contours)

        print(f"Segmentation results saved in: {output_dir}")


    def run_middle_segementation(self,sam_model_path,image_path,show=False):

        """
        Image path is a folder
        
        """
        seg= Segmenter()

        seg.segment_pictures_with_input_prompt(sam_model_path,image_path,show=show)


        return