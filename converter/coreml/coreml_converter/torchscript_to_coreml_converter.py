import logging

import coremltools as ct
from PIL import Image
from coremltools.proto import Model_pb2
from helpers.constants import NORMALIZATION_FACTOR, GREEN, BLUE, RED, END_COLOR, IMAGE_NAME


class TorchscriptToRawCoreMLConverter:
    """ Class that converts a TorchScript to CoreML

    Attributes
    ----------
    model: ModelWrapper
        The model to convert to CoreML
    """

    def __init__(self, pt_model):
        self.pt_model = pt_model

    def convert(self) -> Model_pb2:
        """ Converts a torchscript to a raw coreml model

        Returns
        ----------
        nn_spec: Model_pb2
            The specification of the CoreML model
        """
        try:
            logging.info(f'{BLUE}Starting CoreML conversion with coremltools {ct.__version__}...{END_COLOR}')
            nn_spec = ct.convert(
                self.pt_model.torchscript_model,
                inputs=[
                    ct.ImageType(name=IMAGE_NAME, shape=self.pt_model.input_shape, scale=1 / NORMALIZATION_FACTOR,
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

        model = ct.models.MLModel(nn_spec)
        try:
            logging.info(f'{BLUE}Dry run...{END_COLOR}')
            img = Image.new('RGB', (
                self.pt_model.model_parameters.input_resolution, self.pt_model.model_parameters.input_resolution))
            output = model.predict({IMAGE_NAME: img})
            predictions_shape = output['all_predictions'].shape
            protos_shape = output['segmentation_protos'].shape
            logging.info(f'{GREEN}Success! Predictions: {predictions_shape}, Protos: {protos_shape} {END_COLOR}')
        except Exception as e:
            raise Exception(f'{RED}Could not use the model for inference:{END_COLOR} {e}')

        # Adds the correct output shapes and data types to the coreml model
        # (1, 25200, 117), (1, 32, 160, 160) Yolov5
        # (1, 116, 8400), (1, 32, 160, 160) Ultralytics
        nn_spec.description.output[0].type.multiArrayType.shape[:] = predictions_shape
        nn_spec.description.output[
            0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

        nn_spec.description.output[1].type.multiArrayType.shape[:] = protos_shape
        nn_spec.description.output[
            1].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

        return nn_spec
