# disabling PyLint docstring, no-self-using
# pylint: disable=C0114,C0115,C0116,R0201
import tempfile

from pathlib import Path
from unittest.case import TestCase

from src.converter.filesystem_util import FileSystemUtil


class YoloHelperTest(TestCase):

    def test_folder_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            print('created temporary directory', tmp_dir)
            output_path = Path(tmp_dir) / 'output'
            test_items = [
                output_path / 'images' / 'train',
                output_path / 'images' / 'val',
                output_path / 'labels' / 'train',
                output_path / 'labels' / 'val',
            ]

            FileSystemUtil.create_yolo_folder_structure(output_path)
            for test_item in test_items:
                self.assertTrue(test_item.exists())
                self.assertTrue(test_item.is_dir())
            # even better: does it contain more folders/files than needed?

    def test_folder_structure_can_be_created_multiple_times(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            print('created temporary directory', tmp_dir)
            output_path = Path(tmp_dir) / 'output'
            test_items = [
                output_path / 'images' / 'train',
                output_path / 'images' / 'val',
                output_path / 'labels' / 'train',
                output_path / 'labels' / 'val',
            ]

            FileSystemUtil.create_yolo_folder_structure(output_path)
            FileSystemUtil.create_yolo_folder_structure(output_path)

            for test_item in test_items:
                self.assertTrue(test_item.exists())
                self.assertTrue(test_item.is_dir())
            # even better: does it contain more folders/files than needed?
