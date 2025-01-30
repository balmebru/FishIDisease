# FishIDisease
Functions and models to Identify diseased fish and their affliction

The main functionality is divided into several parts:

Preprocessing: Prepare input images for analysis.
Fish Identification and Segmentation: Detect and isolate the fish from the background.
Lighting Correction: Adjust for lighting conditions using a reference object to ensure robust segmentation and disease identification.
Disease Localization: Identify and highlight diseased areas within the segmented fish masks.
The core functionality will be implemented in the FishIDisease class, which orchestrates these steps. The pipeline takes low-framerate video footage from aquaculture environments as input and outputs comprehensive information about the health status of the fish. Definitions for specific health metrics and disease classification criteria are yet to be finalized.

# Preprocessor


Ideas and popular methods for preprocessing the image are

- Background Subtraction: Remove non-relevant background elements to isolate the region of interest (ROI).
- Selective Denoising: Apply a median filter with a 5x5 kernel for noise reduction while preserving edges. (Reference: Jin, L., & Liang, H. (2017). Deep learning for underwater image recognition in small sample size situations. OCEANS 2017 - Aberdeen, 1–4. https://doi.org/10.1109/OCEANSE.2017.8084645)
- Color and Lighting Correction: Balance uneven lighting and correct color shifts for consistent data.
- Histogram Equalization: Enhance contrast by redistributing pixel intensity values, useful for improving visibility in low-light images.
- Edge Enhancement: Use techniques like Laplacian filters to highlight boundaries for more precise segmentation.
- Image Normalization: Scale pixel values to a consistent range (e.g., [0, 1] or [-1, 1]) to stabilize model performance.

-> decide which methods to use (high prio)

# ReferenceMaker

Should Detect a Reference Image and Help with Lighting and Color Correction
The input will be a low-framerate video, and the output will provide reference measures for lighting and color (potentially size as well).

- Create YOLOv11 Object Segmenter Instance: Fine-tune the model on the reference object to ensure accurate detection.
- Detect and Segment the Reference Object: Extract the reference object from each frame to isolate it for analysis.
- Determine Correction Factors:
    - Lighting Correction: Compute the average pixel intensity of the reference object and adjust image brightness accordingly.
    - Color Balance: Calculate the RGB channel means from the reference object and normalize the image to achieve consistent color representation.
    - Size Normalization (if needed): Measure the reference object's bounding box dimensions and scale images for consistent object sizes during analysis.

# Identification and Segmentation

This step uses the fine-tuned model to detect and segment fish in each frame. Image validation is performed at this stage.

- Decide if Valid Image: Evaluate whether the frame contains a recognizable fish or sufficient image quality for processing.
- Fish Detection: Identify the fish, potentially using aspect ratio as a heuristic to determine if the fish is "sideways."
- Segmentation:
    - Option 1: Use the YOLOv11 segmenter function directly for both detection and segmentation.
    - Option 2: Apply the YOLO model for detection and SAM2 for segmentation if more complex mask annotations are required.
- Extract Mask: Generate and store the segmentation mask for further analysis.
- Extract Low-Level Features: Measure fish attributes such as redness, size, color patterns, or other relevant indicators for disease detection.
![First_segmentation_mask_with_prompt](https://github.com/user-attachments/assets/1bf02f43-06bf-44a2-871b-c46828498721)

  
# Disease ID 
The main functionality processes the information extracted during the identification and segmentation step to classify fish health status.

- Input Features: Define based on low-level information extracted, such as redness levels, size anomalies, or abnormal color patterns.
- General Scoring: Use a continuous health score (e.g., from 0 to 1) representing overall fish health.
- Granular Classification Ideas:
    - Classify diseases into predefined categories (e.g., fungal infection, bacterial lesion, physical injury).
    - Clustering methods (such as k-means) to identify natural groupings in feature data.
    - Outlier Detection: Flag highly abnormal patterns as potential "unknown" disease categories for manual review.

# High level representation (?)

To make the output actionable, the information needs to be presented in a clear and user-friendly manner.

Information Presentation:
- Visual overlays on segmented fish images (e.g., highlighting diseased areas in red).(?)
- Tabular summaries for batch analysis, including health scores, detected issues, and key attributes like redness levels or size deviations.
- Alerts or flags for critical health concerns.

# Connect sensor information to the Image (?)

Linking sensor data with image analysis can provide context for environmental conditions affecting fish health.

Common Sensors:
- Turbidity: Measures water clarity, which can affect image quality and fish health.
- pH: Detects acidity or alkalinity changes, potentially influencing disease prevalence.
- Dissolved Oxygen: Essential for maintaining fish health.
- Temperature: Key for identifying stress conditions or optimal ianges for aquaculture.
  
Potential Integration:
Synchronize sensor readings with video timestamps to contextualize disease detection.
Use metadata fusion techniques to correlate sensor values with observed health issues.

