# pylint: disable=C0114,C0115,C0116,R0201
from pathlib import Path
import yaml


class YamlUtil:
    """ YAML """

    @staticmethod
    def write_yolo_config_file(train_path: str, val_path: str, names: list, yolo_config_file: Path):
        """ write YoloV5 YAML configuration file 
        INPUT:
            -names: class names"""

        config = {}
        # Optional: config['download'] = '...'
        config['train'] = train_path
        config['val'] = val_path
        config['nc'] = len(names)
        config['names'] = names

        with yolo_config_file.open('w') as f:
            yaml.dump(config, f)
