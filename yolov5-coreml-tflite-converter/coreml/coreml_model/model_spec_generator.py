import logging

import coremltools as ct

from helpers.constants import CONFIDENCE_NAME, COORDINATES_NAME, IMAGE_NAME, IOU_NAME, CONF_NAME, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, \
    BATCH_SIZE, END_COLOR, BLUE


class ModelSpecGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, builderSpec, nmsSpec):
        '''
        Combines the coreml model with export logic and the nms to one final model.
        '''
        logging.info(f'{BLUE}Combining CoreML model with NMS and export model...{END_COLOR}')
        # Combine models to a single one
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                (IMAGE_NAME,
                 ct.models.datatypes.Array(BATCH_SIZE, self.model.input_resolution, self.model.input_resolution)),
                (IOU_NAME, ct.models.datatypes.Double()),
                (CONF_NAME, ct.models.datatypes.Double())
            ],
            output_features=[CONFIDENCE_NAME, COORDINATES_NAME]
        )

        # Required version (>= ios13) in order for nms to work
        pipeline.spec.specificationVersion = 4

        pipeline.add_model(builderSpec)
        pipeline.add_model(nmsSpec)

        pipeline.spec.description.input[0].ParseFromString(
            builderSpec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(
            nmsSpec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(
            nmsSpec.description.output[1].SerializeToString())

        # Metadata for the model‚
        pipeline.spec.description.input[
            1].shortDescription = f"(optional) IOU Threshold override (Default: {DEFAULT_IOU_THRESHOLD})"
        pipeline.spec.description.input[
            2].shortDescription = f"(optional) Confidence Threshold override (Default: {DEFAULT_CONF_THRESHOLD})"
        pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
        pipeline.spec.description.output[
            1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

        return pipeline
