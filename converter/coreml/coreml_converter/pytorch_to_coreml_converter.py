import logging
from pathlib import Path

import coremltools as ct
from PIL import Image
from coreml_model.coreml_segmentation_model_spec import CoreMLSegmentationModelSpec
from coremltools.models.model import MLModel
from helpers.constants import BLUE, END_COLOR, FLOAT16_SUFFIX, INT8_SUFFIX, FLOAT16, INT8, SEGMENTATION, IMAGE_NAME, \
    IOU_NAME, CONF_NAME, \
    DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, COREML
from helpers.converter import Converter
from helpers.parameters import ModelParameters, ConversionParameters
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.model import Model


class PytorchToCoreMLConverter(Converter):
    """ Class that converts a Pytorch model to a CoreML model

    Attributes
    ----------
    model_input_path: String
        The path to the Pytorch model

    model_output_path: String
        The path to the converted model

    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. type, use nms, use normalization, max detections ...)

    conversion_parameters: ConversionParameters
        The parameters for the conversion (e.g. quantization types, ...)

    overwrite: bool
        Whether to overwrite existing converted model
    """

    def __init__(self, model_input_path: str,
                 model_output_path: str = None,
                 model_parameters: ModelParameters = ModelParameters(),
                 conversion_parameters: ConversionParameters = ConversionParameters(),
                 overwrite: bool = False):
        super().__init__(model_input_path=model_input_path,
                         model_output_path=model_output_path,
                         model_parameters=model_parameters,
                         conversion_parameters=conversion_parameters,
                         overwrite=overwrite, destination=COREML)

    def _convert(self):
        # For segmentation, we need to add the export layer
        if self.model_parameters.model_type == SEGMENTATION:
            # Get CoreML spec
            model_spec = CoreMLSegmentationModelSpec(self.pt_model).generate_specs()
            model = ct.models.MLModel(model_spec.spec)
        # For detection, we can directly export it with NMS
        else:
            logging.info(f'{BLUE}Starting CoreML export...{END_COLOR}')
            exporter = Exporter(overrides={"format": "coreml", "nms": True})
            converted_model_path = Model(self.model_input_path).export(format="coreml", nms=True)

            logging.info(f'{BLUE}Considering {converted_model_path} for quantization...{END_COLOR}')
            model = MLModel(converted_model_path)
            logging.info(f'{BLUE}Deleting {converted_model_path}...{END_COLOR}')
            Path(converted_model_path).unlink()

        spec = model.get_spec()
        self.inputs = [input.name for input in spec.description.input]
        self.outputs = [output.name for output in spec.description.output]
        return model

    def _dry_run(self, model):
        img = Image.new('RGB', (
            self.pt_model.model_parameters.input_resolution, self.pt_model.model_parameters.input_resolution))
        try:
            output = model.predict(
                {IMAGE_NAME: img, IOU_NAME: DEFAULT_IOU_THRESHOLD, CONF_NAME: DEFAULT_CONF_THRESHOLD})
        except:
            output = model.predict(
                {IMAGE_NAME: img, IOU_NAME: [DEFAULT_IOU_THRESHOLD], CONF_NAME: [DEFAULT_CONF_THRESHOLD]})

    def _export_with_quantization(self, model, quantization_type):
        converted_model = model
        if quantization_type == FLOAT16:
            suffix = FLOAT16_SUFFIX
            converted_model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16)
        elif quantization_type == INT8:
            suffix = INT8_SUFFIX
            converted_model = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8)
        return converted_model

    def _set_metadata(self, model, quantization_type):
        # Set the model metadata
        metadata = self.pt_model.model_parameters.metadata.copy()
        model.short_description = metadata.pop('description')
        model.author = metadata.pop('author')
        model.license = metadata.pop('license')
        model.version = metadata.pop('version')
        model.user_defined_metadata.update({'input_names': str(self.inputs),
                                            'output_names': str(self.outputs),
                                            'quantization_type': quantization_type})
        model.user_defined_metadata.update({k: str(v) for k, v in metadata.items()})

    def _save_model(self, converted_model, converted_model_path):
        converted_model.save(converted_model_path)

    def _on_end(self):
        pass
