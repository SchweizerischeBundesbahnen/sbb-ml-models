import logging

import numpy as np
import onnx
import onnxruntime
import torch

from helpers.constants import BATCH_SIZE, FLOAT16, IMAGE_NAME, SCORES_NAME, CLASSES_NAME, \
    BOUNDINGBOX_NAME, SIMPLE, PREDICTIONS_NAME
from tf_model.tf_nms import NMS
from tf_utils.parameters import PostprocessingParameters


class ONNXModel:
    """ Class to load a ONNX model and run inference

        Attributes
        ----------
        model_path: str
            The path to the ONNX model
        """
    def __init__(self, model_path):
        logging.info("- Initializing ONNX model...")
        self.model = onnx.load(model_path)
        self.session = onnxruntime.InferenceSession(model_path)

        # Load labels and information from metadata
        self.labels, self.quantized, self.normalized, self.nms = self.__read_from_metadata()
        # Get input size
        input_shape = self.session.get_inputs()[0].shape

        self.batch_size = int(input_shape[0])
        self.img_size = (int(input_shape[2]), int(input_shape[3]))

        logging.info(f"- There are {len(self.labels)} labels.")

    def get_input_info(self):
        """
        Returns the information about the input

        Returns
        ----------
        normalized, img size, batch size, PIL image, channel first
            Information about the input
        """
        return not self.normalized, self.img_size, BATCH_SIZE, False, True  # Normalized, img size, batch size, PIL image, channel first

    def get_labels(self):
        return self.labels

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
        if self.quantized == FLOAT16:
            img = np.array(img, dtype=np.float16)

        if self.nms:
            # NMS is included in the model
            outputs = self.session.run(None, {IMAGE_NAME: np.array(img)})
            yxyx = outputs[0]
            classes = outputs[1]
            scores = outputs[2]
            nb_detected = classes.shape[1]

            return yxyx, classes, scores, None, torch.Tensor([nb_detected])

        else:
            predictions = self.session.run([PREDICTIONS_NAME], {IMAGE_NAME: np.array(img)})[0]

            nms = NMS(PostprocessingParameters(nms_type=SIMPLE), [iou_threshold], [conf_threshold])
            yxyx, classes, scores, nb_detected = nms.compute(predictions)

            return yxyx, classes, scores, None, nb_detected

    def __read_from_metadata(self):
        metadata = self.model.metadata_props
        labels = None
        quantized = None
        normalized = None
        nms = None
        for m in metadata:
            if m.key == 'labels':
                labels = m.value.split(',')
            if m.key == 'quantized':
                quantized = m.value
            if m.key == 'do_normalize':
                normalized = True if m.value == 'True' else False
            if m.key == 'do_nms':
                nms = True if m.value == 'True' else False
        if labels is not None and quantized is not None and normalized is not None:
            return labels, quantized, normalized, nms
        else:
            raise ValueError("The model lacks some information (labels or quantized).")
