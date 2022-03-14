# pylint: disable=C0114,C0115,C0116,R0201
from pathlib import Path
import kwcoco


class FileSystemUtil:

    @staticmethod
    def create_yolo_folder_structure(root_folder: Path):
        """ create file structure for YoloV5 """
        subfolders = ['images', 'labels']
        splits = ['train', 'val']
        for subfolder in subfolders:
            for split in splits:
                (root_folder / subfolder / split).mkdir(parents=True, exist_ok=True)

        return root_folder / subfolders[0] / splits[0], root_folder / subfolders[0] / splits[1], root_folder / subfolders[1] / splits[0], root_folder / subfolders[1] / splits[1]
