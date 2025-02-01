

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


    def view_segmentation_SAM_bbox_format(self,image_dir_path, mask_dir_path, output_dir_path):
        # Ensure the output directory exists
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        # Iterate over images and corresponding mask files
        for image_name, mask_name in zip(os.listdir(image_dir_path), os.listdir(mask_dir_path)):
            image_path = os.path.join(image_dir_path, image_name)
            mask_path = os.path.join(mask_dir_path, mask_name)
            output_path = os.path.join(output_dir_path, image_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image {image_path}. Skipping...")
                continue

            # Read the bounding box information from the mask file
            with open(mask_path, 'r') as file:
                lines = file.readlines()

            # Parse each line to get the bounding box coordinates
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convert YOLO format to OpenCV format
                img_height, img_width = image.shape[:2]
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # Draw the bounding box on the image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Save the image with bounding boxes to the output directory
            cv2.imwrite(output_path, image)

        return

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
            predictions, segmentation_masks, results = seg_instace.predict_image(image_path, yolo_model_path,save_image_path=save_image_path,show=show)


        return 

    def predict_image(self,image_path, yolo_model_path, show):

        seg_instace = Segmenter()

        predictions, segmentation_masks,results = seg_instace.predict_image(image_path, yolo_model_path,show)

        return predictions, segmentation_masks,results


    def create_overlay_fish_and_eye(self, image_dir, yolo_fish_segmenter_path, yolo_eye_detector_path, save_path, show=True):
        # Load models
        yolo_eye_detector = YOLO(yolo_eye_detector_path)
        yolo_fish_segmenter = YOLO(yolo_fish_segmenter_path)

        # Create save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Process each image in the directory
        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            save_image_path = os.path.join(save_path, image)

            # Load the image
            img = cv2.imread(image_path)
            # resize the image to 640x480

            img_resized = cv2.resize(img, (640, 480))
            if img is None:
                print(f"Error loading image: {image_path}")
                continue

            # Perform fish segmentation
            result_fish = yolo_fish_segmenter.predict(image_path)
            for r in result_fish:
                if r.masks is not None:
                    for mask in r.masks:
                        # Extract the segmentation mask
                        mask = mask.data[0].cpu().numpy()
                        mask = (mask > 0.5).astype(np.uint8)  # Binary mask

                        # Create a color mask (e.g., green)
                        color_mask = np.zeros_like(img_resized)
                        color_mask[mask == 1] = [0, 255, 0]  # Green color

                        # Blend the mask with the image
                        img = cv2.addWeighted(img_resized, 1, color_mask, 0.5, 0)

            

            result_eye = yolo_eye_detector.predict(img_resized)
            for r in result_eye:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Extract bounding box coordinates
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        img_height, img_width = img.shape[:2]
                        print("Image dimensions:", img_width, img_height)

                        # Extract class ID and confidence
                        class_id = int(box.cls[0].item())
                        confidence = box.conf[0].item()

                        # Draw bounding box
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                        # Put class label and confidence
                        label = f"Eye {class_id} {confidence:.2f}"
                        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Save the resulting image
            cv2.imwrite(save_image_path, img)

            # Optionally display the image
            if show:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return
    

    def create_comparison_plot_prediction_and_image(self, prediction_dir, image_dir, output_dir):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get list of images in both directories
        prediction_images = sorted(os.listdir(prediction_dir))
        original_images = sorted(os.listdir(image_dir))

        # Ensure the number of images in both directories is the same
        if len(prediction_images) != len(original_images):
            print("Warning: The number of images in the prediction and image directories do not match.")
            return

        # Iterate through each pair of images
        for pred_image_name, orig_image_name in zip(prediction_images, original_images):
            # Load images
            pred_image_path = os.path.join(prediction_dir, pred_image_name)
            orig_image_path = os.path.join(image_dir, orig_image_name)

            pred_image = cv2.imread(pred_image_path)
            orig_image = cv2.imread(orig_image_path)

            # Convert BGR to RGB (OpenCV loads images in BGR format)
            pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # Create a figure and subplots
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Plot original image
            axes[0].imshow(orig_image)
            axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
            axes[0].axis('off')  # Hide axes

            # Plot prediction image
            axes[1].imshow(pred_image)
            axes[1].set_title("Prediction Image", fontsize=12, fontweight='bold')
            axes[1].axis('off')  # Hide axes

            # Add a super title for the entire figure
            fig.suptitle(f"Comparison: {orig_image_name}", fontsize=14, fontweight='bold', y=1.02)

            # Adjust layout for better spacing
            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(output_dir, f"comparison_{orig_image_name}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to free memory

        print(f"Comparison plots saved to: {output_dir}")