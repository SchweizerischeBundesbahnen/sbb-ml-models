import logging

import coremltools as ct
from coremltools.models.pipeline import Pipeline
from helpers.constants import CONFIDENCE_NAME, COORDINATES_NAME, IMAGE_NAME, IOU_NAME, CONF_NAME, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, \
    BATCH_SIZE, END_COLOR, BLUE, MASKS_NAME, NUMBER_NAME


class ModelSpecGenerator:
    """ Class that creates the final specifications

    Attributes
    ----------
    model: ModelWrapper
        The model to be converted to CoreML
    """

    def __init__(self, model):
        self.model = model

    def generate(self, model1_spec, model2_spec) -> Pipeline:
        """ Combines the two models specifications into one pipeline

        Parameters
        ----------
        model1_spec: Model_pb2
            The model specification for the first part of the model

        model2_spec: Model_pb2
            The model specification for the first part of the model

        Returns
        ----------
        pipeline: Pipeline
            The pipeline of models to be executed sequentially.

        """
        logging.info(f'{BLUE}Combining CoreML model with NMS and export model...{END_COLOR}')
        # Implement NMS: e.g. IoU and conf need to be an array
        inputs = [
            (IMAGE_NAME,
             ct.models.datatypes.Array(BATCH_SIZE, self.model.model_parameters.input_resolution,
                                       self.model.model_parameters.input_resolution)),
            (IOU_NAME, ct.models.datatypes.Array(1, )),
            (CONF_NAME, ct.models.datatypes.Array(1, ))
        ]
        # confidence score for each predictions/classes, coordinates and masks for each predictions, number of detections
        outputs = [CONFIDENCE_NAME, COORDINATES_NAME, MASKS_NAME, NUMBER_NAME]

        pipeline = ct.models.pipeline.Pipeline(
            input_features=inputs,
            output_features=outputs
        )

        # Required version (>= ios13) in order for nms to work
        pipeline.spec.specificationVersion = 4

        pipeline.add_model(model1_spec)
        pipeline.add_model(model2_spec)

        pipeline.spec.description.input[0].ParseFromString(
            model1_spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(
            model2_spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(
            model2_spec.description.output[1].SerializeToString())

        # Metadata for the model
        pipeline.spec.description.input[
            1].shortDescription = f"IOU Threshold override (Default: {DEFAULT_IOU_THRESHOLD})"
        pipeline.spec.description.input[
            2].shortDescription = f"Confidence Threshold override (Default: {DEFAULT_CONF_THRESHOLD})"
        pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
        pipeline.spec.description.output[
            1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

        pipeline.spec.description.output[2].ParseFromString(
            model2_spec.description.output[2].SerializeToString())
        pipeline.spec.description.output[2].shortDescription = u"Boxes \xd7 [masks height, masks width]"
        pipeline.spec.description.output[3].ParseFromString(
            model2_spec.description.output[3].SerializeToString())
        pipeline.spec.description.output[3].shortDescription = u"Number of detections"
        return pipeline
