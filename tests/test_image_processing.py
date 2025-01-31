import os
import unittest
import cv2
import matplotlib.pyplot as plt
from unittest.mock import patch
from Fishdisease.Preprocessor import Preprocessor

file_path = os.path.join('assets', 'images', 'LINDA', 'EelisaHackathonDatasetsLinda', 'FishDiseaseZHAW', 'BleedingVSBloodCirculation')
image_path = os.path.join(file_path, "ZHAW Biocam_00_20230530085510.jpg")

def load_image(image_path):
    """Load an image from the specified path."""
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load the image. Check the file path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_images(original_image, processed_image):
    """Display the original and processed images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(processed_image)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")

    plt.show()

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image_path = image_path
        self.preprocessor = Preprocessor()

    def test_load_image(self):
        try:
            image = load_image(self.image_path)
            self.assertIsNotNone(image, "Image should not be None")
            print("Image loaded successfully!")
        except FileNotFoundError as e:
            self.fail(f"FileNotFoundError raised: {e}")

    @patch('Fishdisease.ReferenceMaker.apply', autospec=True)
    def test_preprocessing(self, mock_apply):
        # Mock the ReferenceMaker.apply method to accept the self and image arguments
        mock_apply.side_effect = lambda self, image: image

        image = load_image(self.image_path)
        processed_image = self.preprocessor.preprocessing(image)
        self.assertIsNotNone(processed_image, "Processed image should not be None")
        display_images(image, processed_image)

if __name__ == "__main__":
    unittest.main()