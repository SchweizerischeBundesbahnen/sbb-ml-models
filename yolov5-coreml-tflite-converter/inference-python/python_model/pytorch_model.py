import logging

import torch

from helpers.constants import BATCH_SIZE, DEFAULT_INPUT_RESOLUTION
from helpers.coordinates import pt_yxyx2xyxy
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from pytorch_utils.pytorch_nms import YoloNMS
from utils.segment.general import process_mask


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

        (yxyx, classes, scores, masks), protos = model(img)

        nb_detected = classes.shape[1]

        xyxy = pt_yxyx2xyxy(yxyx[0][:nb_detected])

        if masks.shape[2] != 0:
            masks = process_mask(protos[0], masks, xyxy, self.img_size, upsample=True)  # HWC
        else:
            masks = None

        yxyx /= self.img_size[0]

        return yxyx, classes, scores, masks, torch.Tensor([nb_detected])
