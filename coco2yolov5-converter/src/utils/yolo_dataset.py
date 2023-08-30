# pylint: disable=C0114,C0115,C0116,R0201
import argparse
import kwcoco
import shutil
import os
import sys
from pathlib import Path
from random import random
import logging

from utils.coordinates_util import convertRelCoco2Yolo, convertAbsCoco2Yolo
from utils.txt_util import TxtUtil
from utils.yaml_util import YamlUtil
from utils.constants import *
import shutil

class YoloDataset:
    def __init__(self, root_folder: Path, overwrite: bool = False, dataset_split_pivot: float = 0.8):
        output_path = Path(root_folder)

        # Check if already exists
        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                logging.info(
                    f"\033[36mThe output dataset already exists ({root_folder}): not doing it again.\033[0m")
                exit(0)
        self.path = output_path
        self.dataset_split_pivot = dataset_split_pivot
        self.image_train, self.image_val, self.label_train, self.label_val = self.__create_yolo_folder_structure()

    def __create_yolo_folder_structure(self):
        """ Create file structure for YoloV5 """
        subfolders = ['images', 'labels']
        splits = ['train', 'val']
        for subfolder in subfolders:
            for split in splits:
                (self.path / subfolder / split).mkdir(parents=True, exist_ok=True)
        return self.path / subfolders[0] / splits[0], self.path / subfolders[0] / splits[1], self.path / \
               subfolders[1] / splits[0], self.path / subfolders[1] / splits[1]

    def writeYaml(self, labels, config_file, new_format: bool):
        YamlUtil.write_yolo_config_file(
            str(self.image_train), str(self.image_val), labels,
            self.path / config_file, new_format)

    def create(self, coco, images_basedir, valid_labels, labels_to_fusion):
        nb_train = 0
        nb_val = 0
        nb_ignored = 0
        # Consider all images
        for image_id in coco.imgs:
            file_name = coco.imgs[image_id]["file_name"]
            aids = coco.index.gid_to_aids[image_id]
            image_path = images_basedir / file_name
            # Get the labels
            yolo_labels = []
            for aid in aids:
                coco_category_id = coco.anns[aid]["category_id"]
                annotation_name = coco.cats[coco_category_id]["name"]
                if 'segmentation' in coco.anns[aid].keys():
                    # Masks for segmentation
                    self.__convert_coco_for_segmentation(coco, aid, annotation_name, yolo_labels, valid_labels, labels_to_fusion)
                else:
                    # Bboxes for detection
                    self.__convert_coco_for_detection(coco, aid, annotation_name, image_id, yolo_labels, valid_labels, labels_to_fusion)

            if len(yolo_labels) != 0:
                images_out_folder, labels_out_folder, nb_train, nb_val = self.__get_out_folders(nb_train, nb_val)
                self.__copy_image(file_name, image_path, images_out_folder)
                self.__write_label(file_name, labels_out_folder, yolo_labels)
            else:
                nb_ignored += 1
                print(f"{file_name} has no annotations.")

        logging.info(f"- {nb_train} images are copied into train")
        logging.info(f"- {nb_val} images are copied into val")
        logging.info(f"- {nb_ignored} images are ignored")

    def __convert_coco_for_detection(self, coco, aid, annotation_name, image_id, yolo_labels, valid_labels, labels_to_fusion):
        img_width = coco.imgs[image_id].get("width", 0)  # optional
        img_height = coco.imgs[image_id].get("height", 0)  # optional
        x, y, w, h = coco.anns[aid]["bbox"]
        if img_height == 0.0 and img_width == 0.0:
            (x_yolo, y_yolo, w_yolo, h_yolo) = convertRelCoco2Yolo(x, y, w, h)
        else:
            (x_yolo, y_yolo, w_yolo, h_yolo) = convertAbsCoco2Yolo(
                x, y, w, h, img_width, img_height)
        annotation_index = labels.index(annotation_name)
        yolo_labels.append([annotation_index, x_yolo, y_yolo, w_yolo, h_yolo])

    def __convert_coco_for_segmentation(self, coco, aid, annotation_name, yolo_labels, valid_labels, labels_to_fusion):
        masks = coco.anns[aid]["segmentation"]
        label = None
        if annotation_name in labels_to_fusion.keys():
            label = labels_to_fusion[annotation_name]
        elif annotation_name in valid_labels:
            label = annotation_name
        if label:
            annotation_index = valid_labels.index(label)
            yolo_labels.append([x for x in [annotation_index] + masks[0]])

    def __copy_image(self, file_name, image_path, images_out_folder):
        # TODO: use symlink instead?
        try:
            os.makedirs(os.path.dirname(images_out_folder / file_name), exist_ok=True)
            shutil.copyfile(image_path, images_out_folder / file_name)
        except FileNotFoundError:
            logging.info(f"\033[31mConversion error:\033[0m the image ({image_path}) does not exist.")
            exit(1)

    def __write_label(self, file_name, labels_out_folder, yolo_labels):
        txt_file_name = Path(file_name).with_suffix(".txt")
        os.makedirs(os.path.dirname(labels_out_folder / txt_file_name), exist_ok=True)
        TxtUtil.write_labels_txt_file(
            yolo_labels, labels_out_folder / txt_file_name)

    def __get_out_folders(self, nb_train, nb_val):
        # Determine whether to add it to train or val
        if random() < self.dataset_split_pivot:
            images_out_folder = self.image_train
            labels_out_folder = self.label_train
            nb_train += 1
        else:
            images_out_folder = self.image_val
            labels_out_folder = self.label_val
            nb_val += 1
        return images_out_folder, labels_out_folder, nb_train, nb_val
