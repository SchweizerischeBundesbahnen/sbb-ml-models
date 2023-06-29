import logging
import time
from typing import List

import tensorflow as tf
from tf_metadata.metadata_writer import MetadataWriter
from tf_utils.io_order import IOOrder
from tf_utils.parameters import ModelParameters, ConversionParameters, PostprocessingParameters
from tf_utils.representative_dataset import RepresentativeDatasetGenerator

from helpers.constants import DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, FULLINT8, INT8, FLOAT16, FLOAT32, COMBINED, \
    BLUE, END_COLOR, RED, GREEN


class KerasToTFLiteConverter:
    """ Class used to convert a Keras model to a TFlite model.

    Attributes
    ----------
    keras_model: tf.keras.Model
        The path to a log file, if set the logs will be written to that file.

    labels: List[str]
        The id of the run, used to make the logger unique.

    tflite_model_paths: List[str]
        The paths of the converted TFlite models

    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. type, use nms, ...)

    conversion_parameters: ConversionParameters
        The parameters for the conversion (e.g. quantization types, ...)

    postprocessing_parameters: PostprocessingParameters
        The parameters for the postprocesssing (if any) (e.g. nms type, ...)

    iou_threshold: float
        The IoU threshold, if hard-coded into the converted model

    conf_threshold: float
        The confidence threshold, if hard-coded into the converted model

    use_representative_dataset: bool
        Whether to use a representative dataset, if quantized
    """

    def __init__(self, keras_model: tf.keras.Model, labels: List[str], tflite_model_paths: List[str],
                 model_parameters: ModelParameters = ModelParameters(),
                 conversion_parameters: ConversionParameters = ConversionParameters(),
                 postprocessing_parameters: PostprocessingParameters = PostprocessingParameters(),
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                 conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                 use_representative_dataset: bool = True):
        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.quantization_types = self.conversion_parameters.quantization_types
        self.postprocessing_parameters = postprocessing_parameters
        self.keras_model = keras_model
        self.labels = labels
        self.tflite_model_paths = tflite_model_paths
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def convert(self):
        """ Converts Keras model to TFLite model and saves it """
        for quantization_type, tflite_model_path in zip(self.quantization_types, self.tflite_model_paths):
            try:
                logging.info(f"{BLUE}Exporting model to TFLite {quantization_type}...{END_COLOR}")
                start_time = time.time()
                self.converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)

                if self.model_parameters.include_nms and quantization_type == FULLINT8:
                    raise ValueError(
                        f"{RED}TFLite export failure:{END_COLOR} The model cannot be quantized fully in int8, it needs float32 fallback for NMS as some operations are not supported.")

                if not self.model_parameters.include_nms and self.model_parameters.include_threshold:
                    raise ValueError(
                        f"{RED}TFLite export failure:{END_COLOR} The model without NMS does not need to have the thresholds as inputs. Flag --include-threshold can only be set when --include-nms is too.")

                # Convert model to TFLite
                self.__set_converter_flags(self.converter, quantization_type)
                tflite_model = self.converter.convert()

                with tflite_model_path.open('wb') as f:
                    f.write(tflite_model)

                # Write metadata
                if self.conversion_parameters.write_metadata:
                    self.__write_metadata(quantization_type, str(tflite_model_path))
                end_time = time.time()
                logging.info(
                    f"{GREEN}TFLite export success:{END_COLOR} it took {int(end_time - start_time)} seconds to export the model to TFLite ({tflite_model_path})")
            except Exception as e:
                raise Exception(f"{RED}TFLite export failure:{END_COLOR} {e}")

    def __write_metadata(self, quantization_type: str, tflite_model_path: str):
        # Write metadata
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        io_order = IOOrder(self.model_parameters)
        input_order = io_order.get_input_order(interpreter)
        output_order = io_order.get_output_order(interpreter)
        metadata_writer = MetadataWriter(input_order, output_order, self.model_parameters,
                                         self.labels, self.postprocessing_parameters.max_det,
                                         quantization_type, tflite_model_path)
        metadata_writer.write()

    def __set_converter_flags(self, converter: tf.lite.TFLiteConverter, quantization_type: str):
        # Prepare the converted by settings the flags
        self.__maybe_add_representative_dataset(converter, quantization_type)
        self.__add_supported_ops(converter)

        if quantization_type == FLOAT16:
            self.__add_float16_conversion_flags(converter)
        elif quantization_type == INT8:
            self.__add_int8_conversion_flags(converter)
        elif quantization_type == FULLINT8:
            self.__add_fullyint8_conversion_flags(converter)

        converter.allow_custom_ops = False
        converter.experimental_new_converter = True

    def __maybe_add_representative_dataset(self, converter: tf.lite.TFLiteConverter, quantization_type: str):
        # Add representative dataset if any
        if quantization_type != FLOAT32 and self.conversion_parameters.use_representative_dataset:
            representative_dataset = RepresentativeDatasetGenerator(self.model_parameters, self.conversion_parameters,
                                                                    self.iou_threshold, self.conf_threshold)
            converter.representative_dataset = representative_dataset.generate()

    def __add_supported_ops(self, converter: tf.lite.TFLiteConverter):
        # Add the supported operations
        if self.model_parameters.include_nms and self.postprocessing_parameters.nms_type == COMBINED:
            # NMS combined needs select ops
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        else:
            # Otherwise builtins ops are enough
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    @staticmethod
    def __add_float16_conversion_flags(converter: tf.lite.TFLiteConverter):
        # Add the float16 conversion flag
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    @staticmethod
    def __add_int8_conversion_flags(converter: tf.lite.TFLiteConverter):
        # Add the int8 conversion flag
        # We use int8 quantization with float32 fallback.
        # TensorFlow export failure: Quantization not yet supported for op: 'TOPK_V2'.
        # Quantization not yet supported for op: 'CAST'.
        # Quantization not yet supported for op: 'NON_MAX_SUPPRESSION_V4'.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_quantizer = False

    @staticmethod
    def __add_fullyint8_conversion_flags(converter: tf.lite.TFLiteConverter):
        # Add the int8 conversion flag
        # Fully quantized - works without NMS
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_quantizer = False
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
