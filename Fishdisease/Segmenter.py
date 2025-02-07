
import os
from ultralytics.data.annotator import auto_annotate, YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt



class Segmenter:
    def __init__(self):
        pass


    def autoannotate_fish(self,data,detection_model,sam_model,output_dir_fish):

        """
        Takes a input directory with images and runs the autoannotate function on the images
        
        
        """
        auto_annotate(data=data,det_model=detection_model,sam_model=sam_model,output_dir=output_dir_fish,conf=0.35,iou=0.9,max_det=1)

        return
    
    

    def segment_pictures_with_input_prompt(self, sam_model_path: str, image_path: str, input_prompt = [0.5, 0.6],show=False):
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

            # check if the image a jpeg otherwise continue
            if image_full_path.endswith('.jpeg'):
                continue

            predictor.set_image(image)
            height, width, _ = image.shape

            # Define prompt coordinates relative to image dimensions (here in the middle)
            input_point = np.array([[input_prompt[0] * width, input_prompt[1] * height]])
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
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contours = [largest_contour]
            if show:
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    self.show_mask(mask, plt.gca())
                    self.show_points(input_point, input_label, plt.gca())
                    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                    plt.axis('off')
                    plt.show()  

            # Save contours to YOLOv8 format
            output_file_path = os.path.splitext(image_full_path)[0] + ".txt"
            self.save_sam_to_yolov8_format(contours, (height, width), output_file_path)
            # Save binary mask
            binary_mask_path = os.path.splitext(image_full_path)[0] + "_mask.npy"
            self.save_binary_mask(mask, binary_mask_path)



    def fish_eye_autoannotate_with_SAM(self, image_path: str, sam_model_path: str, yolo_model_path: str, show=False, save_path=None,x=0.03,y=0.4):
        """
        This function uses the YOLOv11 segmentation model to identify fish in images. It draws a
        bounding box around the fish, then uses x,y coordinates to pass the prompt to the SAM model
        to segment the fish. The x,y coordinates are approximately at the location where a fish eye
        is usually located.

        Args:
            image_path (str): Path to the input image.
            sam_model_path (str): Path to the SAM model checkpoint.
            yolo_model_path (str): Path to the YOLOv8 model checkpoint.
            show (bool): Whether to display the results.
        """
        
        # Load the YOLOv8 model
        yolo_model = YOLO(yolo_model_path)
        
        # Load the SAM model
        sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        sam_predictor = SamPredictor(sam)
        
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv11 to get fish bounding boxes
        results = yolo_model(image_rgb)
        
        # only keep the highest confidence detection
        results = [result[0] for result in results]



        # Iterate over each detection
        for result in results:
            # Access the boxes object
            boxes = result.boxes
            # Extract bounding box coordinates, confidence, and class IDs
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates in xyxy format
                conf = box.conf[0].item()  # Get confidence score
                cls = box.cls[0].item()  # Get class ID
                
                # Estimate the eye location (top-left corner of the bounding box)
                eye_x = x1 + (x2 - x1) * x  # 5% from the left edge of the bbox
                eye_y = y1 + (y2 - y1) * y  # 30% from the top edge of the bbox
                
                # Convert eye location to relative coordinates (normalized to [0, 1])
                image_height, image_width, _ = image.shape
                eye_x_rel = eye_x / image_width
                eye_y_rel = eye_y / image_height
                
                # Prepare the input prompt for SAM
                input_point = np.array([[eye_x, eye_y]])  # SAM expects absolute coordinates
                input_label = np.array([1])  # 1 indicates a foreground point
                
                # Run SAM to segment the fish
                sam_predictor.set_image(image_rgb)
                masks, scores, _ = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                
                # Get the best mask (highest score)
                best_mask = masks[np.argmax(scores)]
                size_of_mask = best_mask.sum()
                check_value = 0.01 * image_height * image_width
                # if the mask is bigger than 10 percent of the image size request a input prompt form the user by
                # showing the image and then save the click location as the input prompt
                if size_of_mask > check_value:
                    input_point = self.get_new_input_point_by_user(image)
                    input_label = np.array([1])  # 1 indicates a foreground point
                    masks, scores, _ = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                    best_mask = masks[np.argmax(scores)]
                # Optionally display the results
                if show:
                    # Draw the bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # also draw the segmentation mask
                    color_mask = np.zeros_like(image)   
                    color_mask[best_mask > 0] = [0, 255, 0]
                    image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)


                    # Draw the estimated eye location
                    cv2.circle(image, (int(eye_x), int(eye_y)), 5, (0, 0, 255), -1)
                    
                    # Overlay the segmentation mask
                    color_mask = np.zeros_like(image)
                    color_mask[best_mask > 0] = [0, 255, 0]  # Green mask
                    image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)
                    
                    # Show the image
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # save the mask in a file
                    # Save contours to YOLOv8 format
                contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    contours = [largest_contour]
                output_file_path = os.path.splitext(image_path)[0] + ".txt"
                self.save_sam_to_yolov8_format(contours, (image_height, image_width), output_file_path)
                # Save binary mask
                binary_mask_path = os.path.splitext(image_path)[0] + "_mask.npy"
                self.save_binary_mask(best_mask, binary_mask_path)

        return best_mask

    def get_new_input_point_by_user(self, image):
        """
        Opens the given image and captures a single user click as the input point.
        The point is returned as a NumPy array suitable for SAM.

        Args:
            image (ndarray): The image in which the user will click.

        Returns:
            np.ndarray: The x, y coordinates of the click in the original image space.
        """
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title("Click on the point you want to select")
        input_point = []

        # Event handler for capturing click
        def on_click(event):
            if event.xdata and event.ydata:
                input_point.append((event.xdata, event.ydata))
                plt.close()  # Close the plot after the first click

        # Connect the click event to the handler
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if not input_point:
            raise ValueError("No point selected by the user.")

        return np.array(input_point)
    

    def predict_image(self, image_path, yolo_model_path, show=False, save_image_path=None):
        """
        Predicts objects in an image using a YOLOv11 model.

        Parameters:
            image_path (str): Path to the input image.
            yolo_model_path (str): Path to the YOLOv11 model file.
            show (bool): Whether to display the segmentation results. Default is False.
            save_path (str): Path to save the image with overlays if specified. Default is None.

        Returns:
            predictions (list): A list of detected objects with their bounding boxes, labels, and confidence scores.
            segmentation_masks (list): A list of segmentation masks for detected objects.
            results: The raw YOLO results instance.
        """
        # Load the YOLOv11 model
        model = YOLO(yolo_model_path)

        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

        # Perform prediction
        results = model.predict(image_path, conf=0.5)

        # Extract bounding boxes, labels, and confidence scores
        predictions = []
        segmentation_masks = []

        for result in results:
            # Extract bounding boxes, labels, and confidence scores
            for box in result.boxes:
                x_center, y_center, box_width, box_height = box.xywh[0].cpu().numpy()

                # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)
                x1 = x_center - (box_width / 2)
                y1 = y_center - (box_height / 2)
                x2 = x_center + (box_width / 2)
                y2 = y_center + (box_height / 2)

                label = int(box.cls)
                confidence = float(box.conf)

                predictions.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': confidence
                })

            # Extract segmentation masks (if available)
            if result.masks is not None:
                for mask in result.masks:
                    segmentation_masks.append(mask.data.cpu().numpy())

        # Plot the image and overlays
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.axis('off')

        # Overlay segmentation masks
        for mask in segmentation_masks:
            ax.imshow(mask, cmap='jet', alpha=0.3)

        # Overlay bounding boxes
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            label = pred['label']
            confidence = pred['confidence']

            # Draw the bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add label and confidence score
            ax.text(x1, y1 - 5, f'Label: {label}, Conf: {confidence:.2f}',
                    color='r', fontsize=8, backgroundcolor='white')

        # Save the overlay image if a path is specified
        if save_image_path:
            fig.savefig(save_image_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)



        return predictions, segmentation_masks, results

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
    
    def show_mask(self,mask, ax, random_color=False):

        """
        Helper function to show a mask on an image. Takes a defined color or a random color.
        Reshapes the mask to the image size and then shows the mask on the image.

        Copied from the SAM documentation.

        Args:
            mask (np.ndarray): Mask to show.
            ax (matplotlib.axes.Axes): Matplotlib axis object.
            random_color (bool): Whether to use a random color for the mask. Default is False.
        
        """

        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self,coords, labels, ax, marker_size=375):
        """
        Helper function for SAM to show points on an image.
        
        Copied from the SAM documentation.

        Args:
            coords (np.ndarray): Array of point coordinates.
            labels (np.ndarray): Array of point labels.
            ax (matplotlib.axes.Axes): Matplotlib axis object.
            marker_size (int): Size of the marker.
        
        """
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

    
    def save_binary_mask(self, mask: np.ndarray, output_path: str) -> None:
        """
        Save the binary mask as a NumPy file.

        Args:
            mask (np.ndarray): Binary mask to save.
            output_path (str): Path to save the binary mask.
        """
        np.save(output_path, mask.astype(np.uint8))