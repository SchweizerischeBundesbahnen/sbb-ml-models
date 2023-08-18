import logging
from abc import ABC, abstractmethod

from helpers.constants import END_COLOR, DEFAULT_INPUT_RESOLUTION, BATCH_SIZE, BOLD


class InferenceModelAbs(ABC):
    do_normalize = False
    do_nms = False
    img_size = (DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION)
    batch_size = BATCH_SIZE
    pil_image = False
    channel_first = False
    labels = []
    model_type = None
    input_dict = {}
    output_dict = {}
    quantization_type = None

    def print_model_info(self):
        """ Print all relevant info about the model """
        logging.info(f"- The model is for {self.model_type} - {self.quantization_type}.")
        logging.info(
            f"- The model takes {self.batch_size} image{'s' if self.batch_size > 1 else ''} of size {self.img_size} at a time.")
        if self.do_normalize:
            logging.info(f"- The input should be normalized [0-1].")
        logging.info(f"- It considers {len(self.labels)} classes.")
        logging.info(
            f"- It has {len(self.input_dict)} input{'s' if len(self.input_dict) > 1 else ''}")
        for k, v in self.input_dict.items():
            logging.info(f"\t{BOLD}{k}{END_COLOR}: {v}")
        logging.info(
            f"- It has {len(self.output_dict)} output{'s' if len(self.output_dict) > 1 else ''}")
        for k, v in self.output_dict.items():
            logging.info(f"\t{BOLD}{k}{END_COLOR}: {v}")
        if self.do_nms:
            logging.info(f"- NMS should be applied to the output (not included in the model).")

    @abstractmethod
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
        pass
