# disabling PyLint docstring, no self using
# pylint: disable=C0114,C0115,C0116,R0201
import tempfile
import yaml
from pathlib import Path

from src.utils.yaml_util import YamlUtil


class TestYamlUtil:
    def test_write_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_config_path = Path(tmp_dir) / 'output'
            YamlUtil.write_yolo_config_file(
                "../train", "../val", ["a", "b", "c", "d"], yaml_config_path, False)
            with yaml_config_path.open('r') as file:
                res = yaml.safe_load(file)
                assert res['train'] == "../train"
                assert res['nc'] == 4  # a,b,c,d

    def test_write_empty_names_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_config_path = Path(tmp_dir) / 'output'
            YamlUtil.write_yolo_config_file(
                "../train", "../val", [], yaml_config_path, False)
            with yaml_config_path.open('r') as file:
                res = yaml.safe_load(file)
                assert res['train'] == "../train"
                assert res['nc'] == 0
