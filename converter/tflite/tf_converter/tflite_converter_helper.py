import tensorflow as tf
from helpers.constants import INT8, FLOAT16, FLOAT32
from tf_utils.representative_dataset import RepresentativeDatasetGenerator
from helpers.parameters import ModelParameters, ConversionParameters

class TFLiteConverterHelper:

    def __init__(self, model_parameters, conversion_parameters):
        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
    def set_converter_flags(self, converter: tf.lite.TFLiteConverter, quantization_type: str):
        # Prepare the converted by settings the flags
        self.__maybe_add_representative_dataset(converter, quantization_type)
        self.__add_supported_ops(converter)

        if quantization_type == FLOAT16:
            self.__add_float16_conversion_flags(converter)
        elif quantization_type == INT8:
            self.__add_int8_conversion_flags(converter)

        converter.allow_custom_ops = False
        converter.experimental_new_converter = True

    def __maybe_add_representative_dataset(self, converter: tf.lite.TFLiteConverter, quantization_type: str):
        # Add representative dataset if any
        if quantization_type != FLOAT32 and self.conversion_parameters.use_representative_dataset:
            representative_dataset = RepresentativeDatasetGenerator(self.model_parameters, self.conversion_parameters)
            converter.representative_dataset = representative_dataset.generate()

    def __add_supported_ops(self, converter: tf.lite.TFLiteConverter):
        # Add the supported operations
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
