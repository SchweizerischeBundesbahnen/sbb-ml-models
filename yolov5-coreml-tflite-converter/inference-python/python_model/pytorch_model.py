import logging

import torch

from constants import BATCH_SIZE, DEFAULT_INPUT_RESOLUTION
from pytorch_loader import PyTorchModelLoader
from pytorch_nms import YoloNMS


class PyTorchModel:
    def __init__(self, model_path, input_resolution=DEFAULT_INPUT_RESOLUTION):
        logging.info("- Initializing PyTorch model...")
        self.model = PyTorchModelLoader(model_path, input_resolution).load(fuse=True)

        # Load labels
        self.labels = self.model.names
        self.img_size = (input_resolution, input_resolution)
        logging.info(f"- There are {len(self.labels)} labels.")

    def get_input_info(self):
        return False, self.img_size, BATCH_SIZE, False, True

    def get_labels(self):
        return self.labels

    def predict(self, img, iou_threshold, conf_threshold):
        model = YoloNMS(self.model, iou_thres=iou_threshold, conf_thres=conf_threshold)

        yxyx, classes, scores = model(img)

        yxyx /= self.img_size[0]
        nb_detected = classes.shape[1]

        return yxyx, classes, scores, torch.Tensor([nb_detected])
