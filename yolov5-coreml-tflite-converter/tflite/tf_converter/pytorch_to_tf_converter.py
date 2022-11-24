import logging
from pathlib import Path

import tensorflow as tf

from helpers.constants import OUTPUT_DIR, DEFAULT_TFLITE_NAME, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FULLINT8_SUFFIX, TFLITE, SAVED_MODEL, \
    FLOAT32, \
    FLOAT16, INT8, FULLINT8, SIMPLE, COMBINED, TFLITE_SUFFIX, END_COLOR, BLUE, GREEN, RED
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from tf_converter.keras_to_tflite_converter import KerasToTFLiteConverter
from tf_model.keras_model import KerasModel
from tf_utils.parameters import ModelParameters, ConversionParameters, PostprocessingParameters

class PytorchToTFConverter:
    # Converts PyTorch model to Keras model, save as saved_model or TFLite
    def __init__(self,
                 model_input_path,
                 model_output_directory=OUTPUT_DIR,
                 model_output_name=DEFAULT_TFLITE_NAME,
                 model_parameters=ModelParameters(),
                 conversion_parameters=ConversionParameters(),
                 postprocessing_parameters=PostprocessingParameters(),
                 iou_threshold=DEFAULT_IOU_THRESHOLD,
                 conf_threshold=DEFAULT_CONF_THRESHOLD):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_directory_path = Path(model_output_directory)
        self.model_out_name = model_output_name

        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.postprocessing_parameters = postprocessing_parameters

        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_directory_path.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        # Load PyTorch model
        self.pt_model = PyTorchModelLoader(self.model_input_path, self.model_parameters.img_size[0]).load(fuse=False)

        self.labels = self.pt_model.names
        self.nb_classes = len(self.labels)
        self.model_parameters.nb_classes = self.nb_classes

    def convert(self, write_metadata=True, use_representative_dataset=False):
        '''
        Converts a PyTorch model to a TF model
        '''
        try:
            if self.conversion_parameters.dest == SAVED_MODEL:
                logging.info(f'{BLUE}Starting TensorFlow saved_model export with TensorFlow {tf.__version__}...{END_COLOR}')
                converted_model_path = self.__convert_saved_model()
                logging.info(f'{GREEN}TensorFlow saved_model export success:{END_COLOR} saved as {converted_model_path}')
            elif self.conversion_parameters.dest == TFLITE:
                logging.info(f'{BLUE}Starting TFLite export with TensorFlow {tf.__version__}...{END_COLOR}')
                converted_model_paths = self.__convert_tflite(write_metadata=write_metadata, use_representative_dataset=use_representative_dataset)
                logging.info(
                    f"{GREEN}TFLite export success:{END_COLOR} saved as {','.join([str(m) for m in converted_model_paths])}")
        except TypeError as e:
            raise TypeError(
                f'{RED}TensorFlow export failure:{END_COLOR} {e}. Please check that the configuration file matches the chosen model.')
        except Exception as e:
            raise Exception(f'{RED}TensorFlow export failure:{END_COLOR} {e}')

    def __create_keras_model(self):
        return KerasModel(model_parameters=self.model_parameters,
                          postprocessing_parameters=self.postprocessing_parameters).create(self.pt_model,
                                                                                           self.iou_threshold,
                                                                                           self.conf_threshold)

    def __convert_saved_model(self):
        keras_model = self.__create_keras_model()

        converted_model_path = self.__get_converted_model_path()
        keras_model.save(converted_model_path, save_format='tf')
        return converted_model_path

    def __convert_tflite(self, write_metadata, use_representative_dataset):
        keras_model = self.__create_keras_model()
        converted_model_paths = self.__get_converted_model_paths(extension=TFLITE_SUFFIX)

        tf_converter = KerasToTFLiteConverter(keras_model, self.labels, converted_model_paths,
                                              model_parameters=self.model_parameters,
                                              conversion_parameters=self.conversion_parameters,
                                              postprocessing_parameters=self.postprocessing_parameters,
                                              iou_threshold=self.iou_threshold,
                                              conf_threshold=self.conf_threshold,
                                              use_representative_dataset=use_representative_dataset)
        tf_converter.convert(write_metadata=write_metadata)
        return converted_model_paths

    def __get_converted_model_paths(self, extension=''):
        model_output_paths = []
        for quantization_type in self.conversion_parameters.quantization_types:
            model_output_path = self.__get_converted_model_path(quantization_type, extension)

            model_output_paths.append(model_output_path)
        return model_output_paths

    def __get_converted_model_path(self, quantization_type='', extension=''):
        basename = self.model_out_name

        if not self.model_parameters.include_normalization:
            basename += '_no-norm'
        if not self.model_parameters.include_nms:
            basename += '_no-nms'
        else:
            if self.postprocessing_parameters.nms_type in [SIMPLE, COMBINED]:
                basename += f'_nms-{self.postprocessing_parameters.nms_type}'
        if self.model_parameters.include_threshold:
            basename += f'_thres-{self.iou_threshold}-{self.conf_threshold}'
        basename += f'_{self.model_parameters.img_size[0]}'

        if quantization_type == INT8:
            basename += INT8_SUFFIX
        elif quantization_type == FLOAT16:
            basename += FLOAT16_SUFFIX
        elif quantization_type == FULLINT8:
            basename += FULLINT8_SUFFIX
        elif quantization_type == FLOAT32:
            basename += FLOAT32_SUFFIX

        return self.model_output_directory_path / Path(basename + extension)
