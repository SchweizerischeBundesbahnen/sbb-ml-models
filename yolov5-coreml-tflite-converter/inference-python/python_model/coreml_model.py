import logging

import torch
from coremltools.models.model import MLModel

from helpers.constants import BATCH_SIZE, IMAGE_NAME, IOU_NAME, CONF_NAME, CONFIDENCE_NAME, COORDINATES_NAME
from pytorch_utils.pytorch_nms import pt_xywh2yxyx_yolo


class CoreMLModel():
    def __init__(self, model_path):
        logging.info("- Initializing CoreML model...")
        self.model = MLModel(model_path)

        spec = self.model.get_spec()
        pipeline = spec.pipeline
        self.labels = [label.rstrip() for label in pipeline.models[-1].nonMaximumSuppression.stringClassLabels.vector]
        logging.info(f"- There are {len(self.labels)} labels.")

        input_details = spec.description.input
        output_details = spec.description.output

        input_names = ', '.join([input.name for input in input_details])
        output_name = ', '.join([output.name for output in output_details])

        self.img_size = (int(input_details[0].type.imageType.height), int(input_details[0].type.imageType.width))
        logging.info(
            f"- The model takes {len(input_details)} input{'s' if len(input_details) > 1 else ''}: {input_names}.")
        logging.info(f"- The image is of size {self.img_size}.")
        logging.info(f"- It has {len(output_details)} output{'s' if len(output_details) > 1 else ''}: {output_name}")

    def predict(self, img, iou_threshold, conf_threshold):
        predictions = self.model.predict(
            {IMAGE_NAME: img, IOU_NAME: iou_threshold, CONF_NAME: conf_threshold})

        yxyx = pt_xywh2yxyx_yolo(predictions[COORDINATES_NAME])  # coordinates are xywh
        classes = torch.argmax(torch.from_numpy(predictions[CONFIDENCE_NAME]), dim=1)
        scores = torch.max(torch.from_numpy(predictions[CONFIDENCE_NAME]), dim=1).values
        nb_detected = predictions[COORDINATES_NAME].shape[0]

        return yxyx.unsqueeze(0), classes.unsqueeze(0), scores.unsqueeze(0), None, torch.Tensor([nb_detected])

    def get_labels(self):
        return self.labels

    def get_input_info(self):
        return False, self.img_size, BATCH_SIZE, True, False  # Normalized, img size, batch size, PIL image, channel first
