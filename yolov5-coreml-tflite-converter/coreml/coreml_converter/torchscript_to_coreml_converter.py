import logging

import coremltools as ct

from constants import IMAGE_NAME, NB_OUTPUTS, NORMALIZATION_FACTOR, BATCH_SIZE, END_COLOR, BLUE, GREEN, RED


class TorchscriptToRawCoreMLConverter:
    def __init__(self, model):
        self.model = model

    def convert(self):
        '''
        Converts a torchscript to a raw coreml model
        '''
        try:
            logging.info(f'{BLUE}Starting CoreML conversion with coremltools {ct.__version__}...{END_COLOR}')
            nn_spec = ct.convert(
                self.model.torchscript_model,
                inputs=[
                    ct.ImageType(name=IMAGE_NAME, shape=self.model.input_shape, scale=1 / NORMALIZATION_FACTOR,
                                 bias=[0, 0, 0])
                ]
            ).get_spec()

            logging.info(f'{GREEN}CoreML conversion success{END_COLOR}')
        except Exception as e:
            raise Exception(f'{RED}CoreML conversion failure:{END_COLOR} {e}')

        '''
        Adds the correct output shapes and data types to the coreml model
        e.g. (1, 3, 80, 80, 51), (1, 3, 40, 40, 51), (1, 3, 20, 20, 51)
        '''
        for i, feature_map_dimension in enumerate(self.model.feature_map_dimensions):
            nn_spec.description.output[i].type.multiArrayType.shape.append(BATCH_SIZE)
            nn_spec.description.output[i].type.multiArrayType.shape.append(len(self.model.anchors))
            nn_spec.description.output[i].type.multiArrayType.shape.append(
                feature_map_dimension)
            nn_spec.description.output[i].type.multiArrayType.shape.append(
                feature_map_dimension)
            # pc, bx, by, bh, bw, c (no of class class labels)
            nn_spec.description.output[i].type.multiArrayType.shape.append(
                self.model.number_of_classes() + NB_OUTPUTS
            )
            nn_spec.description.output[
                i].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

        return nn_spec
