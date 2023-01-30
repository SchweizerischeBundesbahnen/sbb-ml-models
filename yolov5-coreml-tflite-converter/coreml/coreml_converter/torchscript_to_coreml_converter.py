import logging

import coremltools as ct
from coremltools.proto import Model_pb2

from helpers.constants import IMAGE_NAME, NB_OUTPUTS, NORMALIZATION_FACTOR, BATCH_SIZE, END_COLOR, BLUE, GREEN, RED, \
    DETECTION


class TorchscriptToRawCoreMLConverter:
    """ Class convert a TorchScript to CoremL

    Attributes
    ----------
    model: ModelWrapper
        The model to convert to CoreML
    """

    def __init__(self, model):
        self.model = model

    def convert(self) -> Model_pb2:
        """ Converts a torchscript to a raw coreml model

        Returns
        ----------
        nn_spec: Model_pb2
            The specification of the CoreML model
        """
        try:
            logging.info(f'{BLUE}Starting CoreML conversion with coremltools {ct.__version__}...{END_COLOR}')
            if self.model.model_type == DETECTION:
                nn_spec = ct.convert(
                    self.model.torchscript_model,
                    inputs=[
                        ct.ImageType(name=IMAGE_NAME, shape=self.model.input_shape, scale=1 / NORMALIZATION_FACTOR,
                                     bias=[0, 0, 0])
                    ]
                ).get_spec()
            else:
                nn_spec = ct.convert(
                    self.model.torchscript_model,
                    inputs=[
                        ct.ImageType(name=IMAGE_NAME, shape=self.model.input_shape, scale=1 / NORMALIZATION_FACTOR,
                                     bias=[0, 0, 0])
                    ],
                    outputs=[
                        ct.TensorType(name="all_predictions"),
                        ct.TensorType(name="segmentation_protos")
                    ]
                ).get_spec()

            logging.info(f'{GREEN}CoreML conversion success{END_COLOR}')
        except Exception as e:
            raise Exception(f'{RED}CoreML conversion failure:{END_COLOR} {e}')

        # Adds the correct output shapes and data types to the coreml model
        if self.model.model_type == DETECTION:
            # (1, 3, 80, 80, 51), (1, 3, 40, 40, 51), (1, 3, 20, 20, 51)
            for i, feature_map_dimension in enumerate(self.model.feature_map_dimensions):
                nn_spec.description.output[i].type.multiArrayType.shape.append(BATCH_SIZE)
                nn_spec.description.output[i].type.multiArrayType.shape.append(len(self.model.anchors))
                nn_spec.description.output[i].type.multiArrayType.shape.append(
                    feature_map_dimension)
                nn_spec.description.output[i].type.multiArrayType.shape.append(
                    feature_map_dimension)
                # pc, bx, by, bh, bw, c (no of class class labels)
                nn_spec.description.output[i].type.multiArrayType.shape.append(
                    self.model.number_of_classes + NB_OUTPUTS
                )
                nn_spec.description.output[
                    i].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
        else:
            # (1, 25200, 117), (1, 32, 160, 160)
            nb_anchors = []
            for i in range(len(self.model.anchors)):
                nb_anchors.append(len(self.model.anchors[i]))
            nb_predictions = sum([nb_anchors[i] * (x ** 2) for i, x in enumerate(self.model.feature_map_dimensions)])

            nn_spec.description.output[0].type.multiArrayType.shape.append(BATCH_SIZE)
            nn_spec.description.output[0].type.multiArrayType.shape.append(nb_predictions)
            nn_spec.description.output[0].type.multiArrayType.shape.append(self.model.number_of_classes + 5 + 32)
            nn_spec.description.output[
                0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

            nn_spec.description.output[1].type.multiArrayType.shape.append(BATCH_SIZE)
            nn_spec.description.output[1].type.multiArrayType.shape.append(32)
            nn_spec.description.output[1].type.multiArrayType.shape.append(self.model.input_shape[2] // 4)
            nn_spec.description.output[1].type.multiArrayType.shape.append(self.model.input_shape[3] // 4)
            nn_spec.description.output[
                1].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

        return nn_spec
