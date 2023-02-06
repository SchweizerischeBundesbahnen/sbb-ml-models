import logging

from coremltools.proto import Model_pb2

from helpers.constants import CONFIDENCE_NAME, COORDINATES_NAME, RAW_PREFIX, IOU_NAME, CONF_NAME, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, \
    NB_OUTPUTS, BLUE, END_COLOR


class NMSModelSpecGenerator:
    """ Class that creates specifications for a NMS model

    Attributes
    ----------
    model: ModelWrapper
        The model to be converted to CoreML
    """

    def __init__(self, model):
        self.model = model

    def generate(self, builder_spec) -> Model_pb2:
        """ Create a coreml model with nms to filter the results of the model

        Parameters
        ----------
        builder_spec: Model_pb2
            The specifications of the model to which the NMS is added

        Returns
        ----------
        nms_spec: Model_pb2
            The specifications of the model with NMS added

        """
        logging.info(f"{BLUE}Creating CoreML NMS model...{END_COLOR}")
        nms_spec = Model_pb2.Model()
        nms_spec.specificationVersion = 4

        # Define input and outputs of the model
        for i in range(2):
            nnOutput = builder_spec.description.output[i].SerializeToString()

            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(nnOutput)

            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(nnOutput)

        nms_spec.description.output[0].name = CONFIDENCE_NAME
        nms_spec.description.output[1].name = COORDINATES_NAME

        # Define output shape of the model (#classes and #outputs - objectness score)
        outputSizes = [self.model.number_of_classes, NB_OUTPUTS - 1]
        for i in range(len(outputSizes)):
            maType = nms_spec.description.output[i].type.multiArrayType
            # First dimension of both output is the number of boxes, which should be flexible
            maType.shapeRange.sizeRanges.add()
            maType.shapeRange.sizeRanges[0].lowerBound = 0
            maType.shapeRange.sizeRanges[0].upperBound = -1
            # Second dimension is fixed, for "confidence" it's the number of classes, for coordinates it's position (x, y) and size (w, h)
            maType.shapeRange.sizeRanges.add()
            maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
            maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
            del maType.shape[:]

        # Define the model type non maximum supression
        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = RAW_PREFIX + CONFIDENCE_NAME
        nms.coordinatesInputFeatureName = RAW_PREFIX + COORDINATES_NAME
        nms.confidenceOutputFeatureName = CONFIDENCE_NAME
        nms.coordinatesOutputFeatureName = COORDINATES_NAME
        nms.iouThresholdInputFeatureName = IOU_NAME
        nms.confidenceThresholdInputFeatureName = CONF_NAME
        # Some good default values for the two additional inputs, can be overwritten when using the model
        nms.iouThreshold = DEFAULT_IOU_THRESHOLD
        nms.confidenceThreshold = DEFAULT_CONF_THRESHOLD
        nms.stringClassLabels.vector.extend(self.model.class_labels)

        return nms_spec
