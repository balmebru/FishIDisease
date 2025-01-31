import unittest
import os
import cv2
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from matplotlib import pyplot as plt
from Fishdisease.Segmenter import Segmenter

class TestSegmenter(unittest.TestCase):
    def setUp(self):
        # Initialize the Segmenter
        self.segmenter = Segmenter()

        # Create dummy data for testing
        self.data = "dummy_data"
        self.detection_model = "dummy_detection_model"
        self.sam_model = "dummy_sam_model"
        self.output_dir_fish = "dummy_output_dir_fish"
        self.sam_model_path = "dummy_sam_model_path"
        self.image_path = "dummy_image_path"
        self.input_prompt = [0.5, 0.6]
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.contours = [np.array([[10, 10], [20, 20], [30, 30]])]
        self.output_path = "dummy_output_path.txt"
        self.mask = np.zeros((100, 100), dtype=np.uint8)

    @patch('ultralytics.data.annotator.auto_annotate')
    @patch('ultralytics.models.yolo.model.YOLO')
    @patch('torch.load')
    def test_autoannotate_fish(self, mock_torch_load, mock_YOLO, mock_auto_annotate):
        mock_YOLO.return_value = MagicMock()  # Mocking YOLO model
        mock_torch_load.return_value = {}
        self.segmenter.autoannotate_fish(self.data, self.detection_model, self.sam_model, self.output_dir_fish)
        mock_auto_annotate.assert_called_once_with(data=self.data, det_model=self.detection_model, sam_model=self.sam_model, output_dir=self.output_dir_fish, conf=0.35, iou=0.9, max_det=1)

    @patch('os.listdir')
    @patch('cv2.imread')
    @patch('cv2.findContours')
    @patch('cv2.imwrite')
    @patch('segment_anything.sam_model_registry')
    @patch('segment_anything.SamPredictor')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('torch.load')
    def test_segment_pictures_with_input_prompt(self, mock_torch_load, mock_open, mock_SamPredictor, mock_sam_model_registry, mock_imwrite, mock_findContours, mock_imread, mock_listdir):
        # Mock the os.listdir to return dummy files
        mock_listdir.return_value = ['image1.jpg', 'image2.jpg']

        # Mock the cv2.imread to return a dummy image
        mock_imread.return_value = self.image

        # Mock the cv2.findContours to return dummy contours
        mock_findContours.return_value = (self.contours, None)

        # Mock the SAM model and predictor
        mock_sam_model_registry.return_value = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (np.array([self.mask]), [0.9], None)
        mock_SamPredictor.return_value = mock_predictor

        # Mock the state_dict to return a dictionary with expected keys and torch.Tensor objects
        mock_torch_load.return_value = {
            "image_encoder.pos_embed": torch.zeros((1, 1280)),
            "image_encoder.patch_embed.proj.weight": torch.zeros((1280, 3, 16, 16)),
            "image_encoder.patch_embed.proj.bias": torch.zeros(1280),
            "image_encoder.blocks.0.norm1.weight": torch.zeros(1280),
            "image_encoder.blocks.0.norm1.bias": torch.zeros(1280),
            "image_encoder.blocks.0.attn.rel_pos_h": torch.zeros((27, 80)),
            "image_encoder.blocks.0.attn.rel_pos_w": torch.zeros((27, 80)),
            "image_encoder.blocks.0.attn.qkv.weight": torch.zeros((3840, 1280)),
            "image_encoder.blocks.0.attn.qkv.bias": torch.zeros(3840),
            "image_encoder.blocks.0.attn.proj.weight": torch.zeros((1280, 1280)),
            "image_encoder.blocks.0.attn.proj.bias": torch.zeros(1280),
            "image_encoder.blocks.0.norm2.weight": torch.zeros(1280),
            "image_encoder.blocks.0.norm2.bias": torch.zeros(1280),
            "image_encoder.blocks.0.mlp.lin1.weight": torch.zeros((5120, 1280)),
            "image_encoder.blocks.0.mlp.lin1.bias": torch.zeros(5120),
            "image_encoder.blocks.0.mlp.lin2.weight": torch.zeros((1280, 5120)),
            "image_encoder.blocks.0.mlp.lin2.bias": torch.zeros(1280),
            # Add other necessary keys here
        }

        # Load the state_dict with strict=False to ignore missing keys
        with patch.object(mock_sam_model_registry.return_value, 'load_state_dict', wraps=mock_sam_model_registry.return_value.load_state_dict) as mock_load_state_dict:
            self.segmenter.segment_pictures_with_input_prompt(self.sam_model_path, self.image_path, self.input_prompt, show=False)
            mock_load_state_dict.assert_called_once_with(mock_torch_load.return_value, strict=False)

        mock_listdir.assert_called_once_with(self.image_path)
        self.assertEqual(mock_imwrite.call_count, 2)

    def test_save_sam_to_yolov8_format(self):
        self.segmenter.save_sam_to_yolov8_format(self.contours, (100, 100), self.output_path)
        with open(self.output_path, 'r') as file:
            lines = file.readlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("1 "))

    def test_show_mask(self):
        fig, ax = plt.subplots()
        self.segmenter.show_mask(self.mask, ax)
        self.assertTrue(len(ax.images) > 0)

    def test_show_points(self):
        fig, ax = plt.subplots()
        coords = np.array([[50, 50], [60, 60]])
        labels = np.array([1, 0])
        self.segmenter.show_points(coords, labels, ax)
        self.assertTrue(len(ax.collections) > 0)

    def test_save_binary_mask(self):
        binary_mask_path = "dummy_binary_mask.npy"
        self.segmenter.save_binary_mask(self.mask, binary_mask_path)
        loaded_mask = np.load(binary_mask_path)
        np.testing.assert_array_equal(self.mask, loaded_mask)

if __name__ == "__main__":
    unittest.main()