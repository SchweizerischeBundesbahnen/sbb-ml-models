import logging
from pathlib import Path
from typing import List

import coremltools as ct
from PIL import Image
from coreml_model.coreml_segmentation_model_spec import CoreMLSegmentationModelSpec
from coremltools.models.model import MLModel
from helpers.constants import DEFAULT_QUANTIZATION_TYPE, \
    DEFAULT_INPUT_RESOLUTION, BLUE, END_COLOR, GREEN, RED
from helpers.constants import FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FLOAT32, FLOAT16, INT8, COREML_SUFFIX, \
    DETECTION, PT_SUFFIX, DEFAULT_MAX_NUMBER_DETECTION, SEGMENTATION, IMAGE_NAME, IOU_NAME, CONF_NAME, \
    DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from tf_utils.parameters import ModelParameters
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.model import Model


class PytorchToCoreMLConverter:
    """ Class that converts a Pytorch model to a Keras model, saved as saved_model or TFLite

    Attributes
    ----------
    model_input_path: String
        The path to the Pytorch model

    model_output_path: String
        The path to the converted model

    quantization_types: List[str]
        The quantization types to use for the conversion

    input_resolution: int
        The input resolution of the image

    max_det: int
        The maximum number of detections that can be made by the CoreML model
    """

    def __init__(self, model_input_path: str,
                 model_output_path: str = None,
                 quantization_types: List[str] = None,
                 input_resolution: int = DEFAULT_INPUT_RESOLUTION,
                 max_det: int = DEFAULT_MAX_NUMBER_DETECTION):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_path = Path(model_output_path) if model_output_path else Path(
            model_input_path.replace(PT_SUFFIX, ''))

        if quantization_types is None:
            quantization_types = [DEFAULT_QUANTIZATION_TYPE]
        self.quantization_types = quantization_types
        self.input_resolution = input_resolution
        self.max_det = max_det

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        """
        Converts a PyTorch model to a CoreML model and saves it
        """
        # Export the model
        self.output_path = f'{str(self.model_output_path)}_{self.input_resolution}'

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

        self.__dry_run(model)
        self.__export_with_quantization(model)

    def __dry_run(self, model):
        try:
            logging.info(f'{BLUE}Dry run...{END_COLOR}')
            img = Image.new('RGB', (
                self.pt_model.model_parameters.input_resolution, self.pt_model.model_parameters.input_resolution))
            try:
                output = model.predict(
                    {IMAGE_NAME: img, IOU_NAME: DEFAULT_IOU_THRESHOLD, CONF_NAME: DEFAULT_CONF_THRESHOLD})
            except:
                output = model.predict(
                    {IMAGE_NAME: img, IOU_NAME: [DEFAULT_IOU_THRESHOLD], CONF_NAME: [DEFAULT_CONF_THRESHOLD]})
            logging.info(f'{GREEN}Success!{END_COLOR}')
        except Exception as e:
            raise Exception(f'{RED}Could not use the model for inference:{END_COLOR} {e}')

    def __set_metadata(self, model, inputs, outputs, quantization_type):
        # Set the model metadata
        metadata = self.pt_model.model_parameters.metadata.copy()
        model.short_description = metadata.pop('description')
        model.author = metadata.pop('author')
        model.license = metadata.pop('license')
        model.version = metadata.pop('version')
        model.user_defined_metadata.update({'input_names': str(inputs),
                         'output_names': str(outputs),
                         'quantization_type': quantization_type})
        model.user_defined_metadata.update({k: str(v) for k, v in metadata.items()})

    def __export_with_quantization(self, model):
        spec = model.get_spec()
        inputs = [input.name for input in spec.description.input]
        outputs = [output.name for output in spec.description.output]
        for quantization_type in self.quantization_types:
            try:
                logging.info(f'{BLUE}Exporting model to CoreML {quantization_type}{END_COLOR}')
                model_temp = model
                if quantization_type == FLOAT32:
                    suffix = FLOAT32_SUFFIX
                elif quantization_type == FLOAT16:
                    suffix = FLOAT16_SUFFIX
                    model_temp = ct.models.neural_network.quantization_utils.quantize_weights(
                        model, nbits=16)
                elif quantization_type == INT8:
                    suffix = INT8_SUFFIX
                    model_temp = ct.models.neural_network.quantization_utils.quantize_weights(
                        model, nbits=8)
                else:
                    raise ValueError(f"The quantization type '{quantization_type}' is not supported.")
                self.__set_metadata(model_temp, inputs, outputs, quantization_type)
                converted_model_path = Path(self.output_path + suffix + COREML_SUFFIX)
                model_temp.save(converted_model_path)
                logging.info(f'{GREEN}CoreML export success:{END_COLOR} saved as {converted_model_path}')
            except Exception as e:
                raise Exception(f'{RED}CoreML export failure: {e}{END_COLOR}')

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        # Load the model
        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.model_parameters = ModelParameters(input_resolution=self.input_resolution)
        self.pt_model = PyTorchModelLoader(self.model_input_path, self.model_parameters).load()
        logging.info(f"{BLUE}\t- The model has {self.model_parameters.nb_classes} classes{END_COLOR}")
        logging.info(
            f"{BLUE}\t- The model is for {self.model_parameters.model_type} ({self.model_parameters.model_orig}){END_COLOR}")
