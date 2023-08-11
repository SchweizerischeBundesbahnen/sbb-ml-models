import logging
import ast
import torch
from coremltools.models.model import MLModel

from helpers.constants import BATCH_SIZE, IMAGE_NAME, IOU_NAME, CONF_NAME, CONFIDENCE_NAME, COORDINATES_NAME, \
    MASKS_NAME, NUMBER_NAME
from pytorch_utils.pytorch_nms import pt_xywh2yxyx_yolo


class CoreMLModel:
    """ Class to load a CoreML model and run inference

    Attributes
    ----------
    model_path: str
        The path to the CoremL model
    """
    def __init__(self, model_path):
        logging.info("- Initializing CoreML model...")
        self.model = MLModel(model_path)

        spec = self.model.get_spec()
        pipeline = spec.pipeline
        self.labels = [label.rstrip() for label in pipeline.models[-1].nonMaximumSuppression.stringClassLabels.vector]
        if len(self.labels) == 0:
            self.labels = ast.literal_eval(self.model.user_defined_metadata['names'])

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

    def get_input_info(self):
        """
        Returns the information about the input

        Returns
        ----------
        normalized, img size, batch size, PIL image, channel first
            Information about the input
        """
        return False, self.img_size, BATCH_SIZE, True, False

    def predict(self, img, iou_threshold, conf_threshold):
        """
        Runs the inference

        Parameters
        ----------
        img: PIL.Image
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
        try:
            predictions = self.model.predict(
                {IMAGE_NAME: img, IOU_NAME: [iou_threshold], CONF_NAME: [conf_threshold]})
        except:
            predictions = self.model.predict(
                {IMAGE_NAME: img, IOU_NAME: iou_threshold, CONF_NAME: conf_threshold})

        yxyx = torch.from_numpy(predictions[COORDINATES_NAME]).view(-1, 4)
        confidence = torch.from_numpy(predictions[CONFIDENCE_NAME]).view(-1, len(self.labels))

        yxyx = pt_xywh2yxyx_yolo(yxyx)  # coordinates are xywh
        classes = torch.argmax(confidence, dim=1)
        scores = torch.max(confidence, dim=1).values
        nb_detected = confidence.shape[0]

        if MASKS_NAME in predictions.keys():
            masks = torch.from_numpy(predictions[MASKS_NAME]).view(-1, self.img_size[0], self.img_size[1])
            nb_detected = predictions[NUMBER_NAME][0]
        else:
            masks = None

        return yxyx.unsqueeze(0), classes.unsqueeze(0), scores.unsqueeze(0), masks, torch.Tensor([nb_detected])
