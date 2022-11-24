import logging
from pathlib import Path

import coremltools as ct

from helpers.constants import FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FLOAT32, FLOAT16, INT8, COREML_SUFFIX
from helpers.constants import OUTPUT_DIR, DEFAULT_COREML_NAME, DEFAULT_QUANTIZATION_TYPE, \
    DEFAULT_INPUT_RESOLUTION, BLUE, END_COLOR, GREEN, RED
from coreml_converter.torchscript_exporter import TorchscriptExporter
from coreml_converter.torchscript_to_coreml_converter import TorchscriptToRawCoreMLConverter
from coreml_model.coreml_export_layer import CoreMLExportLayerGenerator
from coreml_model.model_spec_generator import ModelSpecGenerator
from coreml_model.nms_model_spec_generator import NMSModelSpecGenerator
from pytorch_utils.pytorch_loader import PyTorchModelLoader


class PytorchToCoreMLConverter:
    def __init__(self, model_input_path, model_output_directory=OUTPUT_DIR,
                 model_output_name=DEFAULT_COREML_NAME,
                 quantization_types=None, input_resolution=DEFAULT_INPUT_RESOLUTION):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_directory_path = Path(model_output_directory)
        self.model_output_name = model_output_name
        if quantization_types is None:
            quantization_types = [DEFAULT_QUANTIZATION_TYPE]
        self.quantization_types = quantization_types
        self.input_resolution = input_resolution

        self.__check_input_create_output()

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_directory_path.mkdir(parents=True, exist_ok=True)

    def convert(self):
        '''
        Converts a PyTorch model to a CoreML model
        '''
        # Load the model
        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.model = PyTorchModelLoader(self.model_input_path, self.input_resolution).load_wrapper()
        self.model.torch_model.model[-1].coreml_export = True

        # Get CoreML spec
        model_spec = self.__get_coreml_spec()

        # Export the model
        file_name = f'{self.model_output_name}_{self.input_resolution}'

        model = ct.models.MLModel(model_spec.spec)
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
                converted_model_path = self.model_output_directory_path / Path(file_name + suffix + COREML_SUFFIX)

                model_temp.save(converted_model_path)
                logging.info(f'{GREEN}CoreML export success:{END_COLOR} saved as {converted_model_path}')
            except Exception as e:
                raise Exception(f'{RED}CoreML export failure: {e}{END_COLOR}')

    def __get_coreml_spec(self):
        # Produces torchscript
        self.model.torchscript_model = TorchscriptExporter(self.model).export()

        # Convert torchscript to raw coreml model
        raw_coreml_model = TorchscriptToRawCoreMLConverter(self.model).convert()

        builder = ct.models.neural_network.NeuralNetworkBuilder(spec=raw_coreml_model)

        # Add export logic to coreml model
        CoreMLExportLayerGenerator(self.model).add_to(builder)

        # Create nms logic
        nms_spec = NMSModelSpecGenerator(self.model).generate(builder.spec)

        # Combine model with export logic and nms logic
        model_spec = ModelSpecGenerator(self.model).generate(builder.spec, nms_spec)

        return model_spec
