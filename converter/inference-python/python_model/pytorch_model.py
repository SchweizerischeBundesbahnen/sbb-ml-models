import logging

import torch

from helpers.constants import BATCH_SIZE, DEFAULT_INPUT_RESOLUTION
from helpers.coordinates import pt_yxyx2xyxy_yolo
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from pytorch_utils.pytorch_nms import YoloNMS
from utils.segment.general import process_mask


class PyTorchModel:
    """ Class to load a PyTorch model and run inference

    Attributes
    ----------
    model_path: str
        The path to the PyTorch model

    input_resolution: int
        The input resolution for the input to the model
    """
    def __init__(self, model_path, input_resolution=DEFAULT_INPUT_RESOLUTION):
        logging.info("- Initializing PyTorch model...")
        self.model = PyTorchModelLoader(model_path, input_resolution).load(fuse=True)

        # Load labels
        self.labels = self.model.class_labels

        self.img_size = (input_resolution, input_resolution)
        logging.info(f"- There are {len(self.labels)} labels.")

    def get_input_info(self):
        """
        Returns the information about the input

        Returns
        ----------
        normalized, img size, batch size, PIL image, channel first
            Information about the input
        """
        return False, self.img_size, BATCH_SIZE, False, True

    def predict(self, img, iou_threshold, conf_threshold):
        """
        Runs the inference

        Parameters
        ----------
        img: torch.Tensor
            The input image

        iou_threshold: float
            The IoU threshold

        conf_threshold: float
            The confidence threshold

        Returns
        ----------
        yxyx, classes, scores, masks, nb_detected
            The detections made by the model
        """
        model = YoloNMS(self.model, iou_thres=iou_threshold, conf_thres=conf_threshold)

        (yxyx, classes, scores, masks), protos = model(img)

        nb_detected = classes.shape[1]

        xyxy = pt_yxyx2xyxy_yolo(yxyx[0][:nb_detected])

        if masks.shape[2] != 0:
            masks = process_mask(protos[0], masks, xyxy, self.img_size, upsample=True)  # HWC
        else:
            masks = None

        yxyx /= self.img_size[0]

        return yxyx, classes, scores, masks, torch.Tensor([nb_detected])
