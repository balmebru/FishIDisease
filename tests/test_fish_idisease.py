import unittest
import os
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from Fishdisease.FishIDisease import FishIDisease
from Fishdisease.Segmenter import Segmenter

class TestFishDisease(unittest.TestCase):
    def setUp(self):
        # Initialize the FishIDisease
        self.fish_disease = FishIDisease()

        # Create dummy data for testing
        self.data = "dummy_data"
        self.detection_model = "dummy_detection_model"
        self.sam_model = "dummy_sam_model"
        self.output_dir_fish = "dummy_output_dir_fish"
        self.image_dir = "dummy_image_dir"
        self.segmentation_dir = "dummy_segmentation_dir"
        self.sam_model_path = "dummy_sam_model_path"
        self.image_path = "dummy_image_path"

    @patch.object(Segmenter, 'autoannotate_fish')
    def test_autoannotate_fish_directory(self, mock_autoannotate_fish):
        self.fish_disease.autoannotate_fish_directory(self.data, self.detection_model, self.sam_model, self.output_dir_fish)
        mock_autoannotate_fish.assert_called_once_with(data=self.data, detection_model=self.detection_model, sam_model=self.sam_model, output_dir_fish=self.output_dir_fish)

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_view_segmentation_after_autoanno(self, mock_imwrite, mock_imread, mock_listdir, mock_makedirs):
        # Mock the os.listdir to return dummy files
        mock_listdir.side_effect = lambda x: ['image1.jpg', 'image2.jpg'] if x == self.image_dir else ['mask1.txt', 'mask2.txt']

        # Mock the cv2.imread to return a dummy image
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock the open function to return dummy mask data
        with patch('builtins.open', unittest.mock.mock_open(read_data="0 0.1 0.1 0.2 0.2")):
            self.fish_disease.view_segmentation_after_autoanno(self.image_dir, self.segmentation_dir)

        mock_makedirs.assert_called_once_with(os.path.join(self.image_dir, "segmentation_results"), exist_ok=True)
        self.assertEqual(mock_imwrite.call_count, 2)

    @patch.object(Segmenter, 'segment_pictures_with_input_prompt')
    def test_run_middle_segementation(self, mock_segment_pictures_with_input_prompt):
        self.fish_disease.run_middle_segementation(self.sam_model_path, self.image_path, show=False)
        mock_segment_pictures_with_input_prompt.assert_called_once_with(self.sam_model_path, self.image_path, show=False)

if __name__ == "__main__":
    unittest.main()