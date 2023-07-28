# pylint: disable=C0114,C0115,C0116,R0201
import yaml
from pathlib import Path


class YamlUtil:
    """ YAML """

    @staticmethod
    def write_yolo_config_file(train_path: str, val_path: str, names: list, yolo_config_file: Path, new_format: bool):
        """ write YoloV5 YAML configuration file """

        config = {}
        if new_format:
            config['train'] = train_path
            config['val'] = val_path
            # Optional: config['test'] = '...'
            config['names'] = { i: label for i, label in enumerate(names) }
            # Optional: config['download'] = '...'
        else:
            # Optional: config['download'] = '...'
            config['train'] = train_path
            config['val'] = val_path
            config['nc'] = len(names)
            config['names'] = names

        with yolo_config_file.open('w') as f:
            yaml.dump(config, f)
