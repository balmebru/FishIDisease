import unittest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from matplotlib import pyplot as plt
from Fishdisease.ReferenceMaker import ReferenceMaker
from Fishdisease.Segmenter import Segmenter

class TestReferenceMaker(unittest.TestCase):
    @patch('ultralytics.YOLO', autospec=True)
    def setUp(self, mock_YOLO):
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_YOLO.return_value = mock_model
        mock_model.predict.return_value = [MagicMock(masks=np.ones((100, 100), dtype=np.uint8) * 255)]

        # Initialize the ReferenceMaker with a dummy model path
        self.model_path = 'dummy_model_path.pt'
        self.reference_maker = ReferenceMaker(self.model_path)

        # Create a dummy image for testing
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (25, 25), (75, 75), (255, 255, 255), -1)  # Add a white square

    def test_detect_reference_object(self):
        # Mock the Segmenter.segment_fish method to return a mask
        original_segment_fish = Segmenter.segment_fish
        Segmenter.segment_fish = lambda x: np.ones((100, 100), dtype=np.uint8) * 255

        mask = self.reference_maker.detect_reference_object(self.image)
        self.assertIsNotNone(mask, "Mask should not be None")
        self.assertEqual(mask.shape, self.image.shape[:2], "Mask shape should match image shape")

        # Restore the original method
        Segmenter.segment_fish = original_segment_fish

    def test_compute_correction_factors(self):
        # Mock the Segmenter.segment_fish method to return a mask
        original_segment_fish = Segmenter.segment_fish
        Segmenter.segment_fish = lambda x: np.ones((100, 100), dtype=np.uint8) * 255

        # Add your test logic here
        # Example:
        factors = self.reference_maker.compute_correction_factors(self.image)
        self.assertIsNotNone(factors, "Factors should not be None")

        # Restore the original method
        Segmenter.segment_fish = original_segment_fish

    def test_resize_to_reference(self):
        # Add your test logic here
        # Example:
        resized_image = self.reference_maker.resize_to_reference(self.image, (25, 25, 75, 75))
        self.assertIsNotNone(resized_image, "Resized image should not be None")

    def test_apply(self):
        # Mock the Segmenter.segment_fish method to return a mask
        original_segment_fish = Segmenter.segment_fish
        Segmenter.segment_fish = lambda x: np.ones((100, 100), dtype=np.uint8) * 255

        # Add your test logic here
        # Example:
        result = self.reference_maker.apply(self.image)
        self.assertIsNotNone(result, "Result should not be None")

        # Restore the original method
        Segmenter.segment_fish = original_segment_fish

if __name__ == "__main__":
    unittest.main()