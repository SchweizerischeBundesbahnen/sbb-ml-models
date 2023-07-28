# disabling PyLint docstring, no self using
# pylint: disable=C0114,C0115,C0116,R0201
import tempfile
from pathlib import Path

from src.utils.txt_util import TxtUtil


class TestTxtUtil:
    def write_labels_to_temp_dir_and_return_content(self, labels):
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_file_path = Path(tmp_dir) / 'file.txt'
            TxtUtil.write_labels_txt_file(labels, txt_file_path)
            with txt_file_path.open('r') as file:
                res = file.read()
            return res

    def test_write_txt(self):
        labels = [[0, 0.34567, 1.0, 0.99999, 0.0001],
                  [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        assert self.write_labels_to_temp_dir_and_return_content(labels) == """0 0.34567 1.0 0.99999 0.0001
0 0.1 0.2 0.3 0.4 0.5 0.6
"""

    def test_write_empty_txt(self):
        labels = []
        assert self.write_labels_to_temp_dir_and_return_content(labels) == ""
