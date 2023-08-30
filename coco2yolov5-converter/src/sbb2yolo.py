""" Main """
import argparse
import kwcoco
import logging
import os
import shutil
import shutil
import sys
from pathlib import Path
from random import random

from utils.constants import *
from utils.coordinates_util import convertRelCoco2Yolo, convertAbsCoco2Yolo
from utils.txt_util import TxtUtil
from utils.yaml_util import YamlUtil
from utils.yolo_dataset import YoloDataset
from utils.coco_datasets import CocoDatasets


class Coco2YoloDatasetConverter:
    def __init__(self, coco_input_folders: list, yolo_output_folder: str, yolo_config_file: str,
                 dataset_split_pivot: float, new_format: bool = True, overwrite: bool = False):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.coco_input_folders = coco_input_folders
        self.yolo_output_folder = yolo_output_folder.rstrip('/')
        self.yolo_config_file = yolo_config_file
        self.dataset_split_pivot = dataset_split_pivot
        self.new_format = new_format
        self.overwrite = overwrite

    def convert(self):
        """ Convert Coco Annotations from JSON file into YoloV5 Structure """
        yolo_dataset = YoloDataset(self.yolo_output_folder, self.overwrite, self.dataset_split_pivot)
        logging.info("\033[36mStarting conversion from Coco to Yolo format...\033[0m")

        valid_labels, labels_to_fusion, cocos, image_folders = CocoDatasets(self.coco_input_folders).read_datasets()

        # Convert all input coco datasets
        for i, (coco, images_basedir) in enumerate(zip(cocos, image_folders)):
            logging.info(f"Converting dataset ({i + 1}/{len(self.coco_input_folders)})")
            yolo_dataset.create(coco, images_basedir, valid_labels, labels_to_fusion)

        # Write yaml config file
        yolo_dataset.writeYaml(valid_labels, self.yolo_config_file, self.new_format)

        logging.info(f"\033[32mConversion success\033[0m: saved in {self.yolo_output_folder}")

        logging.info(
            f"- The dataset contains {len([x for x in list(Path(yolo_dataset.image_train).glob('**/*')) if x.is_file()])} images in train and {len([x for x in list(Path(yolo_dataset.image_val).glob('**/*')) if x.is_file()])} images in val")
        logging.info(f"- There are {len(valid_labels)} labels")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-input-folders', dest='coco_input_folders', type=str, nargs='+',
                        help='Paths to COCO dataset', required=True)
    parser.add_argument('--yolo-output-folder', dest='yolo_output_folder', type=str, default='output',
                        help='Output folder (where to save Yolo dataset). Default: ./output')
    parser.add_argument('--yolo-config-file', dest='yolo_config_file', type=str,
                        default='config.yaml', help='YoloV5 YAML file name. Default: config.yaml')
    parser.add_argument('--dataset-split-pivot', dest='dataset_split_pivot', type=float, default=0.8,
                        help='Train/val split (0.8 = 80 percent in train, 20 percent in val). Default: 0.8')
    parser.add_argument('--new-format', action='store_true', help="If set, will use the slightly different 'new' format for yolo yaml config.")
    parser.add_argument('--overwrite', action='store_true',
                        help="If set, will overwrite output dataset if it already exist.")
    # TODO 7: add argument include labels
    # TODO 8: add argument exclude labels
    # TODO 9: add argument for move and copy
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    Coco2YoloDatasetConverter(args.coco_input_folders, args.yolo_output_folder,
                              args.yolo_config_file, args.dataset_split_pivot, args.new_format, args.overwrite).convert()
