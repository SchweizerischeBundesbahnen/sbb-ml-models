import tensorflow as tf
from helpers.constants import TFLITE, BATCH_SIZE, NB_CHANNEL, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD
from helpers.converter import Converter
from helpers.parameters import ModelParameters, ConversionParameters
from tf_converter.tflite_converter_helper import TFLiteConverterHelper
from tf_model.keras_model import KerasModel
from tf_utils.io_order import IOOrder
from tf_metadata.metadata_writer import MetadataWriter

class PytorchToTFLiteConverter(Converter):
    """ Class to convert a Pytorch model to TFLite

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
        super().__init__(model_input_path=model_input_path,
                         model_output_path=model_output_path,
                         model_parameters=model_parameters,
                         conversion_parameters=conversion_parameters,
                         overwrite=overwrite, destination=TFLITE)

    def _convert(self):
        # Convert model to Keras
        return KerasModel(model_parameters=self.model_parameters).create(self.pt_model.torch_model,
                                                                         self.model_input_path)

    def _dry_run(self, model):
        img = tf.zeros(
            (BATCH_SIZE, self.model_parameters.input_resolution, self.model_parameters.input_resolution,
             NB_CHANNEL))
        if self.model_parameters.include_nms:
            y = model.predict([img, tf.constant([DEFAULT_IOU_THRESHOLD]), tf.constant([DEFAULT_CONF_THRESHOLD])])
        else:
            y = model.predict(img)

    def _export_with_quantization(self, model, quantization_type):
        # Convert model to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        TFLiteConverterHelper(self.model_parameters, self.conversion_parameters).set_converter_flags(converter, quantization_type)
        tflite_model = converter.convert()
        return tflite_model

    def _save_model(self, converted_model, converted_model_path):
        with converted_model_path.open('wb') as f:
            f.write(converted_model)

    def _set_metadata(self, converted_model_path, quantization_type):
        interpreter = tf.lite.Interpreter(model_path=str(converted_model_path))
        io_order = IOOrder(self.model_parameters)
        input_order = io_order.get_input_order(interpreter)
        output_order = io_order.get_output_order(interpreter)
        metadata_writer = MetadataWriter(input_order, output_order, self.model_parameters,
                                         quantization_type, converted_model_path)
        metadata_writer.write()

    def _on_end(self):
        pass
