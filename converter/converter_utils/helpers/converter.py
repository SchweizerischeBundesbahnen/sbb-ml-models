import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from helpers.constants import FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FLOAT32, FLOAT16, INT8, COREML_SUFFIX, \
    TFLITE_SUFFIX, ONNX_SUFFIX, \
    TFLITE, COREML, ONNX, \
    BLUE, END_COLOR, GREEN, RED, PT_SUFFIX
from helpers.parameters import ModelParameters, ConversionParameters
from pytorch_utils.pytorch_loader import PyTorchModelLoader


class Converter(ABC):
    """ Class that converts a Pytorch model

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
                 overwrite: bool = False,
                 destination: str = TFLITE):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_path = Path(model_output_path) if model_output_path else Path(
            model_input_path.replace(PT_SUFFIX, ''))

        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.overwrite = overwrite

        if destination not in [TFLITE, COREML, ONNX]:
            logging.info(
                f"{BLUE}The format '{destination}' is not supported (should one of '{TFLITE}', '{CORELM}' or '{ONNX}'){END_COLOR}")
            exit(1)
        destinations = {TFLITE: 'TFLite', COREML: 'CoreML', ONNX: 'ONNX'}
        self.destination = destinations[destination]
        suffixes = {TFLITE: TFLITE_SUFFIX, COREML: COREML_SUFFIX, ONNX: ONNX_SUFFIX}
        self.suffix = suffixes[destination]

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        """
        Converts a PyTorch model
        """
        # Get converted model paths
        self.converted_model_paths = self.__get_converted_model_paths()

        if len(self.converted_model_paths) == 0:
            logging.info(f'{BLUE}The model has already been converted (use --overwrite to overwrite it).{END_COLOR}')
            exit(0)

        model = self._convert()
        self.__dry_run(model)
        self.__export_with_quantizations(model)

        self._on_end()

    @abstractmethod
    def _convert(self):
        pass

    def __dry_run(self, model):
        try:
            logging.info(f'{BLUE}Dry run...{END_COLOR}')
            self._dry_run(model)
            logging.info(f'{GREEN}Success!{END_COLOR}')
        except Exception as e:
            raise Exception(f'{RED}Could not use the model for inference:{END_COLOR} {e}')

    @abstractmethod
    def _dry_run(self, model):
        pass

    def __export_with_quantizations(self, model):
        for quantization_type, converted_model_path in zip(self.conversion_parameters.quantization_types,
                                                           self.converted_model_paths):
            try:
                logging.info(f"{BLUE}Exporting model to {self.destination} {quantization_type}...{END_COLOR}")
                start_time = time.time()

                converted_model = self._export_with_quantization(model, quantization_type)

                if self.destination.lower() == TFLITE:
                    self._save_model(converted_model, converted_model_path)
                    if self.conversion_parameters.write_metadata:
                        self._set_metadata(converted_model_path, quantization_type)
                else:
                    self._set_metadata(converted_model, quantization_type)
                    if self.conversion_parameters.write_metadata:
                        self._save_model(converted_model, converted_model_path)

                end_time = time.time()
                logging.info(
                    f"{GREEN}{self.destination} export success:{END_COLOR} it took {int(end_time - start_time)} seconds to export the model to {self.destination} ({converted_model_path})")
            except Exception as e:
                raise Exception(f"{RED}{self.destination} export failure:{END_COLOR} {e}")

    @abstractmethod
    def _export_with_quantization(self, model, quantization_type, converted_model_path):
        pass

    @abstractmethod
    def _set_metadata(self, model, inputs, outputs, quantization_type):
        pass

    @abstractmethod
    def _save_model(self, converted_model):
        pass

    @abstractmethod
    def _on_end(self, converted_model):
        pass

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        # Load the model
        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.pt_model = PyTorchModelLoader(self.model_input_path, self.model_parameters).load()
        logging.info(f"{BLUE}\t- The model has {self.model_parameters.nb_classes} classes{END_COLOR}")
        logging.info(
            f"{BLUE}\t- The model is for {self.model_parameters.model_type} ({self.model_parameters.model_orig}){END_COLOR}")

    def __get_converted_model_paths(self):
        # Get the names of the convert models
        model_output_paths = []
        already_done = []
        for quantization_type in self.conversion_parameters.quantization_types:
            model_output_path = self.__get_converted_model_path(quantization_type)
            if os.path.exists(model_output_path) and not self.overwrite:
                logging.info(
                    f"{BLUE}Converted {self.destination} model ({model_output_path}) already exists.{END_COLOR}")
                already_done.append(quantization_type)
            else:
                model_output_paths.append(model_output_path)
        for ad in already_done:
            self.conversion_parameters.quantization_types.remove(ad)

        return model_output_paths

    def __get_converted_model_path(self, quantization_type: str = ''):
        # Basename (given as input)
        basename = str(self.model_output_path)

        # Specify if it does not include normalization or nms
        if not self.model_parameters.include_normalization:
            basename += '_no-norm'
        if not self.model_parameters.include_nms:
            basename += '_no-nms'

        # Add img size and type
        basename += f'_{self.model_parameters.input_resolution}'

        # Add quantization type
        if quantization_type == INT8:
            basename += INT8_SUFFIX
        elif quantization_type == FLOAT16:
            basename += FLOAT16_SUFFIX
        elif quantization_type == FLOAT32:
            basename += FLOAT32_SUFFIX
        return Path(basename + self.suffix)
