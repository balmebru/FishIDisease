# FishIDisease

![License](https://img.shields.io/badge/License-GNU-blue?style=for-the-badge)
![BeHappyFish](https://img.shields.io/badge/Project-BeHappyFish-purple?style=for-the-badge)
[![VERSION](https://img.shields.io/badge/VERSION-0.1.0-COLOR.svg?style=for-the-badge&logo=LOGO)](<LINK>)
![Build Status](https://img.shields.io/badge/build-failing-red?style=for-the-badge)
![Container Size](https://img.shields.io/badge/Container%20Size-<1GB-blue?style=for-the-badge&logo=docker)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Yolo-0078D7?style=for-the-badge&logo=Yolo&logoColor=white)
![AI](https://img.shields.io/badge/AI-0078D7?style=for-the-badge&logo=AI&logoColor=white)
 

## Build the workspace

Unzip or clone the [project file](https://github.com/balmebru/FishIDisease) 

Ensure that your Docker service is running, then open the project folder with VS Code.

VS Code will propmt you to reopen as a container. Accept and watch the logs on terminal

## Background

Functions and models to Identify diseased fish and their affliction

The main functionality is divided into several parts:

- Preprocessing: Prepare input images for analysis.

- Fish Identification and Segmentation: Detect and isolate the fish from the background.

- Lighting Correction: Adjust for lighting conditions using a reference object to ensure robust segmentation and disease identification.

- Disease Localization: Identify and highlight diseased areas within the segmented fish masks.

The core functionality will be implemented in the FishIDisease class, which orchestrates these steps. The pipeline takes low-framerate video footage from aquaculture environments as input and outputs comprehensive information about the health status of the fish. Definitions for specific health metrics and disease classification criteria are yet to be finalized.


### Preprocessor


Ideas and popular methods for preprocessing the image are

- Background Subtraction: Remove non-relevant background elements to isolate the region of interest (ROI).
- Selective Denoising: Apply a median filter with a 5x5 kernel for noise reduction while preserving edges. (Reference: Jin, L., & Liang, H. (2017). Deep learning for underwater image recognition in small sample size situations. OCEANS 2017 - Aberdeen, 1–4. https://doi.org/10.1109/OCEANSE.2017.8084645)
- Color and Lighting Correction: Balance uneven lighting and correct color shifts for consistent data.
- Histogram Equalization: Enhance contrast by redistributing pixel intensity values, useful for improving visibility in low-light images.
- Edge Enhancement: Use techniques like Laplacian filters to highlight boundaries for more precise segmentation.
- Image Normalization: Scale pixel values to a consistent range (e.g., [0, 1] or [-1, 1]) to stabilize model performance.

-> decide which methods to use (high prio)

### ReferenceMaker

Should Detect a Reference Image and Help with Lighting and Color Correction
The input will be a low-framerate video, and the output will provide reference measures for lighting and color (potentially size as well).

- Create YOLOv11 Object Segmenter Instance: Fine-tune the model on the reference object to ensure accurate detection.
- Detect and Segment the Reference Object: Extract the reference object from each frame to isolate it for analysis.
- Determine Correction Factors:
    - Lighting Correction: Compute the average pixel intensity of the reference object and adjust image brightness accordingly.
    - Color Balance: Calculate the RGB channel means from the reference object and normalize the image to achieve consistent color representation.
    - Size Normalization (if needed): Measure the reference object's bounding box dimensions and scale images for consistent object sizes during analysis.

### Identification and Segmentation

This step uses the fine-tuned model to detect and segment fish in each frame. Image validation is performed at this stage.

- Decide if Valid Image: Evaluate whether the frame contains a recognizable fish or sufficient image quality for processing.
- Fish Detection: Identify the fish, potentially using aspect ratio as a heuristic to determine if the fish is "sideways."
- Segmentation:
    - Option 1: Use the YOLOv11 segmenter function directly for both detection and segmentation.
    - Option 2: Apply the YOLO model for detection and SAM2 for segmentation if more complex mask annotations are required.
- Extract Mask: Generate and store the segmentation mask for further analysis.
- Extract Low-Level Features: Measure fish attributes such as redness, size, color patterns, or other relevant indicators for disease detection.

Below the SAM output and the prompted point:


![First_segmentation_mask_with_prompt](https://github.com/user-attachments/assets/1bf02f43-06bf-44a2-871b-c46828498721)


Below the Eye segmentation with manual imput prompt, the fish got detected by the trained yolov11 segmentaton model and the eye location estimated to segment the eye using SAM:


![fishdetection](https://github.com/user-attachments/assets/fd72210c-c01c-4b90-8d04-1811457d2516)


For the detector of diseased eyes around 10 image per instance (healthy and diseased) got annotated by bounding boxes. This was used to train a yolov11 detection model.


![Eye_Detection_healthy](https://github.com/user-attachments/assets/face8a0b-9d79-4a04-b790-c7fae9d50e61)
![Eye_Detection](https://github.com/user-attachments/assets/9d3c10d8-65ad-43af-b574-5c69bbe40b29)

The model can detect other fish species (but maybe because of the white background) and the eye of other fish. This indicates the model can determine "eyes" correctly. 

![image](https://github.com/user-attachments/assets/e5af6a5e-1827-4308-aaa9-50deefef1474)




A second data set was used to generate the underwater fish detection because the inital model trained on the example dataset (with fishes on white background) could not detect fishes in real underwater videos. The dataset (https://alzayats.github.io/DeepFish/) contains bounding boxes around fishes in underwater videos. There are around 2500 frames from different fish species.

Applying the yolov11 detection model to real underwater videos on which the model was not trained resulted in acceptable preformance (considering different lighting conditions).

![image](https://github.com/user-attachments/assets/70e6a829-8569-4ca9-a29b-65b1526eee39)


The main issue is now two fold:

- Determine the correct detection "depth" for multiple fishes behind each other (possibly applying a binary filter).
- Create a validation step to determine if a picture is even valid
- Create a validation step if a mask is valid

### Disease ID 
The main functionality processes the information extracted during the identification and segmentation step to classify fish health status.

- Input Features: Define based on low-level information extracted, such as redness levels, size anomalies, or abnormal color patterns.
- General Scoring: Use a continuous health score (e.g., from 0 to 1) representing overall fish health.
- Granular Classification Ideas:
    - Classify diseases into predefined categories (e.g., fungal infection, bacterial lesion, physical injury).
    - Clustering methods (such as k-means) to identify natural groupings in feature data.
    - Outlier Detection: Flag highly abnormal patterns as potential "unknown" disease categories for manual review.

### Dataset description

The main data is collected as images from aquaculture in Switzerland. The pictures of the fish are taken as sideways full pictures on a white cutting board. Therefore, it is somewhat likely that the model learns to separate based on the white background.

The dataset contains different types of pictures:

- Total data
    - FishDisease
         - EyeDisease
               - Healthy
               - Diseased
    - Weight
        - Over or Underweight
    - Bleeding vs Blood Circulation
    - FishCount
        - Videos from inside the tank




  

