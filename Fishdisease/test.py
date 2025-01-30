import cv2
import matplotlib.pyplot as plt
from Preprocessor import Preprocessor  # Ensure this file is in the same directory

# Load an image (change the path to match your test image)
image_path = "C:/Users/z005221s/Downloads/ZHAW Biocam_00_20240325095815.jpg"
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error: Could not load the image. Check the file path.")
else:
    print("Image loaded successfully!")

# Convert image from BGR (OpenCV format) to RGB (for proper visualization in matplotlib)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an instance of Preprocessor
preprocessor = Preprocessor()

# Apply preprocessing
processed_image = preprocessor.preprocessing(image)

# Show original and processed images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(processed_image)
axes[1].set_title("Processed Image")
axes[1].axis("off")

plt.show()
