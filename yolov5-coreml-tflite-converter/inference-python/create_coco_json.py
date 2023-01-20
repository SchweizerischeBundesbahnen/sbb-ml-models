import argparse
import json
import logging
from pathlib import Path

import PIL
import cv2
import numpy as np
import torch
from utils.datasets import LoadImages

from constants import RED, DEFAULT_INPUT_RESOLUTION, END_COLOR, BLUE, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    NORMALIZATION_FACTOR
from coordinates import pt_yxyx2xywh_coco, pt_normalize_xywh
from python_model.pytorch_model import PyTorchModel
from python_utils.plots import scale_coords_yolo, plot_boxes_xywh


class DetectionsToCoco:
    def __init__(self, model_path, pt_input_resolution=DEFAULT_INPUT_RESOLUTION):
        self.model_path = model_path
        self.pt_input_resolution = pt_input_resolution

        self.__init_model()

    def __init_model(self):
        self.model_name = Path(self.model_path).name

        if not Path(self.model_path).exists():
            raise ValueError(f"{RED}Model not found:{END_COLOR} {self.model_path}")

        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.prefix = 'pytorch'
        try:
            self.model = PyTorchModel(self.model_path, self.pt_input_resolution)
        except Exception as e:
            raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
        self.do_normalize, self.img_size, self.batch_size, self.pil_image, self.channel_first = self.model.get_input_info()
        self.labels = self.model.get_labels()

    def __init_coco_json(self):
        coco_json = {}
        coco_json['info'] = [str(self.model_path)]
        coco_json['licenses'] = []
        coco_json['categories'] = []
        coco_json['videos'] = []
        coco_json['images'] = []
        coco_json['annotations'] = []
        return coco_json

    def __populate_categories(self, coco_json):
        for i, category in enumerate(self.labels):
            coco_json['categories'].append({"id": i + 1, "name": category})
        return coco_json

    def create(self, img_dir, iou_threshold=DEFAULT_IOU_THRESHOLD,
               conf_threshold=DEFAULT_CONF_THRESHOLD, save_img=False):
        img_path = Path(img_dir)

        if not img_path.exists():
            raise ValueError(f"The directory ({img_dir}) containing the images to predict does not exist.")

        # Loads a single image (no batch)
        dataset = LoadImages(img_dir, img_size=self.img_size, auto=False)

        coco_json = self.__init_coco_json()
        coco_json = self.__populate_categories(coco_json)

        aid = 1
        for i, (img_path, img, img_orig, _) in enumerate(dataset):
            img_name = Path(img_path).name
            coco_json['images'].append({"id": i + 1, "file_name": img_name})

            img = torch.from_numpy(img)
            img = img.float()

            if self.do_normalize:
                # Normalize image
                img = img.float() / NORMALIZATION_FACTOR

            if self.pil_image:
                img = PIL.Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
            else:
                img = img.unsqueeze(0)

            yxyx, classes, scores, nb_detected = self.model.predict(img, iou_threshold, conf_threshold)

            yxyx = scale_coords_yolo(img_orig, self.img_size,
                                     yxyx)  # From rescaled img to orig img (i.e. resized, padding)
            xywh = pt_yxyx2xywh_coco(yxyx)  # yxyx to xywh
            xywh = pt_normalize_xywh(xywh, img_orig)  # normalized

            if save_img:
                plot_boxes_xywh([img_orig], xywh, classes, scores, nb_detected, self.labels)
                # A single image is considered here
                img_annotated = img_orig
                out_path_img = f"../data/output/detections/XYWH_{self.prefix}_{img_name.rsplit('.')[0]}_boxes_{self.model_name.rsplit('.')[0]}.png"
                cv2.imwrite(out_path_img, img_annotated)

            for annotation, label in zip(xywh[0], classes[0]):
                category = self.labels[int(label)]
                category_id = self.labels.index(category) + 1
                coco_json['annotations'].append(
                    {"id": aid, "image_id": i + 1, "category_id": category_id, "bbox": [float(x) for x in annotation]})
                aid += 1

        with Path(img_dir, 'coco.json').open('w') as f:
            json.dump(coco_json, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        required=True,
                        help=f"The path to the converted model (tflite or coreml).")
    parser.add_argument('--img-dir', type=str, required=True,
                        help=f"The path to the images.")
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f'IoU threshold. Default: {DEFAULT_IOU_THRESHOLD}')
    parser.add_argument('--conf-threshold', type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f'Confidence threshold. Default: {DEFAULT_CONF_THRESHOLD}')
    parser.add_argument('--img-size', type=int, default=DEFAULT_INPUT_RESOLUTION,
                        help='For Pytorch model, the input resolution')
    parser.add_argument('--save-img', action='store_true', help="If set, will save the results.")

    opt = parser.parse_args()

    detector = DetectionsToCoco(opt.model, opt.img_size)
    detector.create(img_dir=opt.img_dir, iou_threshold=opt.iou_threshold, conf_threshold=opt.conf_threshold,
                    save_img=opt.save_img)
