import argparse
import kwcoco
import logging
import os
import shutil
import shutil
import sys
from pathlib import Path
from random import random

class CocoDatasets:
    def __init__(self, root_folders):
        self.root_folders = root_folders
        self.labels = []
        self.cocos = []
        self.image_folders = []

    def read_datasets(self):
        labels = []
        data = []
        for i, root_folder in enumerate(self.root_folders):
            logging.info(
                f"Reading from dataset: {root_folder.rstrip('/')} ({i + 1}/{len(self.root_folders)})")
            self.__init_dataset(root_folder)

        return list(dict.fromkeys(self.labels)), self.cocos, self.image_folders

    def __init_dataset(self, root_folder):
        input_path = Path(root_folder)

        if not input_path.exists():
            logging.info(f"\033[31mConversion error:\033[0m the input path does not exist ({input_path})")
            exit(1)
        annotations_files = list(input_path.glob('**/*.json'))
        if not len(annotations_files) == 1:
            logging.info(
                f"\033[31mConversion error:\033[0m annotation files are not unique or are empty: {[str(a) for a in annotations_files]}")
            exit(1)
        annotations_file = annotations_files[0]
        coco = kwcoco.CocoDataset(str(annotations_file))
        annotation_names = [coco.cats[a]["name"] for a in coco.cats]
        self.cocos.append(coco)
        self.image_folders.append(Path(annotations_file).parents[0])
        self.labels.extend(annotation_names)
        logging.info(f"- There are {len(coco.imgs)} images")
        logging.info(f"- There are {len(annotation_names)} labels")
