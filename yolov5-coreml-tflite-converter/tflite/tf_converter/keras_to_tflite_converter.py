import time
import logging
import tensorflow as tf

from constants import DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, FULLINT8, INT8, FLOAT16, FLOAT32, COMBINED, BLUE, \
    END_COLOR, RED, GREEN
from tf_metadata.metadata_writer import MetadataWriter
from tf_utils.io_order import IOOrder
from tf_utils.parameters import ModelParameters, ConversionParameters, PostprocessingParameters
from tf_utils.representative_dataset import RepresentativeDatasetGenerator


class KerasToTFLiteConverter:
    # Converts Keras model to TFlite model
    def __init__(self, keras_model, labels, tflite_model_paths,
                 model_parameters=ModelParameters(),
                 conversion_parameters=ConversionParameters(),
                 postprocessing_parameters=PostprocessingParameters(),
                 iou_threshold=DEFAULT_IOU_THRESHOLD,
                 conf_threshold=DEFAULT_CONF_THRESHOLD,
                 use_representative_dataset=True):

        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.quantization_types = self.conversion_parameters.quantization_types
        self.postprocessing_parameters = postprocessing_parameters
        self.keras_model = keras_model
        self.labels = labels
        self.tflite_model_paths = tflite_model_paths
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.use_representative_dataset = use_representative_dataset

    def convert(self, write_metadata=True):
        '''
        Converts Keras model to TFLite model
        '''
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
                if write_metadata:
                    self.__write_metadata(quantization_type, str(tflite_model_path))
                end_time = time.time()
                logging.info(
                    f"{GREEN}TFLite export success:{END_COLOR} it took {int(end_time - start_time)} seconds to export the model to TFLite ({tflite_model_path})")
            except Exception as e:
                raise Exception(f"{RED}TFLite export failure:{END_COLOR} {e}")

    def __write_metadata(self, quantization_type, tflite_model_path):
        # Write metadata
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        io_order = IOOrder(self.model_parameters)
        input_order = io_order.get_input_order(interpreter)
        output_order = io_order.get_output_order(interpreter)
        metadata_writer = MetadataWriter(input_order, output_order, self.model_parameters,
                                         self.labels, self.postprocessing_parameters.max_det,
                                         quantization_type, tflite_model_path)
        metadata_writer.write()

    def __set_converter_flags(self, converter, quantization_type):
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

    def __maybe_add_representative_dataset(self, converter, quantization_type):
        if quantization_type != FLOAT32 and self.use_representative_dataset:
            representative_dataset = RepresentativeDatasetGenerator(self.model_parameters, self.conversion_parameters,
                                                                    self.iou_threshold, self.conf_threshold)
            converter.representative_dataset = representative_dataset.generate()

    def __add_supported_ops(self, converter):
        # NMS combined needs select ops
        if self.model_parameters.include_nms and self.postprocessing_parameters.nms_type == COMBINED:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    def __add_float16_conversion_flags(self, converter):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    def __add_int8_conversion_flags(self, converter):
        # Int8 quantization with Float32 fallback.
        # TensorFlow export failure: Quantization not yet supported for op: 'TOPK_V2'.
        # Quantization not yet supported for op: 'CAST'.
        # Quantization not yet supported for op: 'NON_MAX_SUPPRESSION_V4'.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_quantizer = False

    def __add_fullyint8_conversion_flags(self, converter):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_quantizer = False
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
