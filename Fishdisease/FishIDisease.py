

from Segmenter import Segmenter
import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from matplotlib import pyplot as plt


class FishIDisease:
    def __init__(self):
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
    

    def video_splitter(self, video_path: str, output_dir: str, frame_rate: int):
        """
        Splits a video into individual frames at the specified frame rate.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory where the extracted frames will be saved.
            frame_rate (int): Number of frames to save per second.

        """
        # Check if output directory exists; create if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the video
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            print("Error: Unable to open video.")
            return

        # Get video properties
        video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, video_fps // frame_rate)
        frame_count = 0

        while True:
            success, frame = video_capture.read()
            if not success:
                break

            # Save the frame if it's on the correct interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved: {frame_filename}")

            frame_count += 1

        video_capture.release()
        print("Video splitting complete.")



    def auto_split_and_move(self, image_dir_path, mask_dir_path, output_dir_path_images, output_dir_path_mask, interval):
        """
        Moves every interval-th image and mask from the input directories to the output directories.
        
        Args:
            image_dir_path (str): Path to the directory containing images.
            mask_dir_path (str): Path to the directory containing masks.
            output_dir_path_images (str): Path to the target directory for images.
            output_dir_path_mask (str): Path to the target directory for masks.
            interval (int): Defines the step size for selecting images and masks.
        """
        # Ensure interval is valid
        if interval <= 0:
            raise ValueError("Interval must be a positive integer.")

        # Ensure output directories exist
        os.makedirs(output_dir_path_images, exist_ok=True)
        os.makedirs(output_dir_path_mask, exist_ok=True)

        # Get sorted lists of images and masks
        image_files = sorted([f for f in os.listdir(image_dir_path) if os.path.isfile(os.path.join(image_dir_path, f))])
        mask_files = sorted([f for f in os.listdir(mask_dir_path) if os.path.isfile(os.path.join(mask_dir_path, f))])

        # Create a set of mask base names for quick lookup
        mask_base_names = {os.path.splitext(f)[0] for f in mask_files}

        # Move files at specified interval
        for idx in range(0, len(image_files), interval):
            image_file = image_files[idx]
            image_base_name = os.path.splitext(image_file)[0]

            # Check for a matching mask by base name
            if image_base_name in mask_base_names:
                corresponding_mask_file = next(f for f in mask_files if os.path.splitext(f)[0] == image_base_name)

                # Paths for input and output
                src_image_path = os.path.join(image_dir_path, image_file)
                src_mask_path = os.path.join(mask_dir_path, corresponding_mask_file)
                dest_image_path = os.path.join(output_dir_path_images, image_file)
                dest_mask_path = os.path.join(output_dir_path_mask, corresponding_mask_file)

                # Move image and mask
                shutil.move(src_image_path, dest_image_path)
                shutil.move(src_mask_path, dest_mask_path)

        print("Files moved successfully.")




    def validate_and_sync_directories(self,dir1, dir2):
        """
        Ensures that both directories have matching base filenames.
        Removes files from either directory if they don't have a corresponding partner.

        Args:
            dir1 (str): Path to the first directory.
            dir2 (str): Path to the second directory.
        """
        # Get base filenames without extensions
        files1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        files2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

        basenames1 = {os.path.splitext(f)[0] for f in files1}
        basenames2 = {os.path.splitext(f)[0] for f in files2}

        # Identify unmatched basenames
        unmatched_in_dir1 = basenames1 - basenames2
        unmatched_in_dir2 = basenames2 - basenames1

        # Remove unmatched files from dir1
        for file in files1:
            basename, _ = os.path.splitext(file)
            if basename in unmatched_in_dir1:
                file_path = os.path.join(dir1, file)
                os.remove(file_path)
                print(f"Removed {file_path} from {dir1}")

        # Remove unmatched files from dir2
        for file in files2:
            basename, _ = os.path.splitext(file)
            if basename in unmatched_in_dir2:
                file_path = os.path.join(dir2, file)
                os.remove(file_path)
                print(f"Removed {file_path} from {dir2}")

        print("Validation and synchronization complete.")



    def autoannotate_fish_eyes_dir(self,image_dir,sam_model_path,yolo_model_path,show=False,save_path=None):

        """
        Uses the autoannotate_fish_eyes function to autoannotate all images in a directory.¨

        """

        for image in os.listdir(image_dir):
            if image.endswith(".jpg") or image.endswith(".png"):
                image_path = os.path.join(image_dir, image)
                self.autoannotate_fish_eyes(image_path,sam_model_path,yolo_model_path,save_path=save_path,show=show)
        return


    def autoannotate_fish_eyes(self, image_path: str, sam_model_path: str, yolo_model_path: str, show=False,save_path=None):

        
        seg_instace = Segmenter()
        best_mask = seg_instace.fish_eye_autoannotate_with_SAM(image_path=image_path, sam_model_path=sam_model_path, yolo_model_path=yolo_model_path, show=show,save_path=save_path)
        return
    

    def predict_directory(self, image_dir, yolo_model_path,save_path, show=True):

        seg_instace = Segmenter()

        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            save_image_path = os.path.join(save_path, image)    
            predictions, segmentation_masks, results = seg_instace.predict_image(image_path, yolo_model_path,save_path=save_image_path,show=show)


        return 

    def predict_image(self,image_path, yolo_model_path, show):

        seg_instace = Segmenter()

        predictions, segmentation_masks,results = seg_instace.predict_image(image_path, yolo_model_path,show)

        return predictions, segmentation_masks,results


