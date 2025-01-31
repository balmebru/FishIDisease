import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from Fishdisease.Preprocessor import Preprocessor
from Fishdisease.ReferenceMaker import ReferenceMaker


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Initialize the Preprocessor
        self.preprocessor = Preprocessor()

        # Create a more complex dummy image for testing
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add a white square
        cv2.rectangle(self.image, (25, 25), (75, 75), (255, 255, 255), -1)

        # Add a red circle
        cv2.circle(self.image, (50, 50), 10, (0, 0, 255), -1)

        # Add some noise
        noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        self.image = cv2.add(self.image, noise)

    def test_background_subtraction(self):
        result = self.preprocessor.background_subtraction(self.image)
        self.assertIsNotNone(result, "Result should not be None")
        self.assertEqual(result.shape, self.image.shape, "Result shape should match image shape")

    def test_apply_denoising(self):
        result = self.preprocessor.apply_denoising(self.image)
        self.assertIsNotNone(result, "Result should not be None")
        self.assertEqual(result.shape, self.image.shape, "Result shape should match image shape")

    @patch('Fishdisease.ReferenceMaker.apply', autospec=True)
    def test_lighting_correction(self, mock_apply):
        # Mock the ReferenceMaker.apply method to return the input image
        mock_apply.return_value = self.image.copy()

        result = self.preprocessor.lighting_correction(self.image)
        self.assertIsNotNone(result, "Result should not be None")
        self.assertEqual(result.shape, self.image.shape, "Result shape should match image shape")

    @patch('Fishdisease.ReferenceMaker.apply', autospec=True)
    def test_preprocessing(self, mock_apply):
        # Mock the ReferenceMaker.apply method to return the input image
        mock_apply.return_value = self.image.copy()

        result = self.preprocessor.preprocessing(self.image)
        self.assertIsNotNone(result, "Result should not be None")
        self.assertEqual(result.shape, self.image.shape, "Result shape should match image shape")

if __name__ == "__main__":
    unittest.main()