import logging
from pathlib import Path

import tensorflow as tf
from helpers.constants import FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FULLINT8_SUFFIX, \
    FLOAT32, FLOAT16, INT8, FULLINT8, TFLITE_SUFFIX, END_COLOR, BLUE, GREEN, RED, PT_SUFFIX
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from tf_converter.keras_to_tflite_converter import KerasToTFLiteConverter
from tf_model.keras_model import KerasModel
from tf_utils.parameters import ModelParameters, ConversionParameters
import os

class PytorchToTFConverter:
    """ Class to convert a Pytorch model to a Keras model, saved as saved_model or TFLite

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

    def __init__(self,
                 model_input_path: str,
                 model_output_path: str = None,
                 model_parameters: ModelParameters = ModelParameters(),
                 conversion_parameters: ConversionParameters = ConversionParameters(),
                 overwrite: bool = False):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_path = Path(model_output_path) if model_output_path else Path(
            model_input_path.replace(PT_SUFFIX, ''))
        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.overwrite = overwrite

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        """ Converts a PyTorch model to a TF model (saved_model or TFLite) and saves it """
        try:
            converted_model_paths = self.__convert_tflite()
            logging.info(
                f"{GREEN}TFLite conversion success:{END_COLOR} saved as {','.join([str(m) for m in converted_model_paths])}")
        except Exception as e:
            raise Exception(f'{RED}TFLite conversion failure:{END_COLOR} {e}')

    def __check_input_create_output(self):
        # Check that the model does exist and create the output directory, if it does not yet exist
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        # Load PyTorch model
        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.pt_loader = PyTorchModelLoader(self.model_input_path, self.model_parameters).load(fuse=False)

        logging.info(f"{BLUE}\t- The model has {self.model_parameters.nb_classes} classes{END_COLOR}")
        logging.info(
            f"{BLUE}\t- The model is for {self.model_parameters.model_type} ({self.model_parameters.model_orig}){END_COLOR}")
        logging.info(
            f"{BLUE}\t- Normalization will{'' if self.model_parameters.include_normalization else ' not'} be included in the model.{END_COLOR}")
        logging.info(
            f"{BLUE}\t- NMS will{'' if self.model_parameters.include_nms else ' not'} be included in the model.{END_COLOR}")

    def __convert_tflite(self):
        converted_model_paths = self.__get_converted_model_paths()

        if len(converted_model_paths) == 0:
            logging.info(f'{BLUE}The model has already been converted (use --overwrite to overwrite it).{END_COLOR}')
            exit(0)

        logging.info(f'{BLUE}Starting TFLite export with TensorFlow {tf.__version__}...{END_COLOR}')
        # Create Keras model
        keras_model = self.__create_keras_model()

        # Convert to TFLite
        tf_converter = KerasToTFLiteConverter(keras_model, converted_model_paths,
                                              model_parameters=self.model_parameters,
                                              conversion_parameters=self.conversion_parameters)
        tf_converter.convert()

        return converted_model_paths

    def __create_keras_model(self):
        # Create the Keras model from the Pytorch one
        return KerasModel(model_parameters=self.model_parameters).create(self.pt_loader.torch_model,
                                                                         self.model_input_path)

    def __get_converted_model_paths(self):
        # Get the names of the convert models
        model_output_paths = []
        already_done = []
        for quantization_type in self.conversion_parameters.quantization_types:
            model_output_path = self.__get_converted_model_path(quantization_type)
            if os.path.exists(model_output_path) and not self.overwrite:
                logging.info(f"{BLUE}Converted model ({model_output_path}) already exists.{END_COLOR}")
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
        elif quantization_type == FULLINT8:
            basename += FULLINT8_SUFFIX
        elif quantization_type == FLOAT32:
            basename += FLOAT32_SUFFIX
        return Path(basename + TFLITE_SUFFIX)
