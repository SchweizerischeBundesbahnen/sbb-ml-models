import os
import unittest

import tensorflow as tf

from constants import DEFAULT_PT_MODEL, OUTPUT_DIR, FLOAT32, FULLINT8, INT8, FLOAT16
from tf_converter.pytorch_to_tf_converter import PytorchToTFConverter
from tf_utils.parameters import ModelParameters, ConversionParameters


class ConverterTest(unittest.TestCase):
    '''
    Checks that the converted model have the right inputs/outputs
    '''

    @staticmethod
    def aux_test(q_type, include_norm, include_nms, include_threshold, input_len, output_len):
        conversion_parameters = ConversionParameters(quantization_types=[q_type], nb_calib=10)
        model_parameters = ModelParameters(include_nms=include_nms, include_normalization=include_norm,
                                           include_threshold=include_threshold)
        converter = PytorchToTFConverter(DEFAULT_PT_MODEL, OUTPUT_DIR,
                                         model_parameters=model_parameters,
                                         conversion_parameters=conversion_parameters)
        # The input/output names are not consistent (i.e. incremental) and therefore not suited for the metadata
        model_files = converter.convert(write_metadata=False)
        # Check that the converted model indeed exists
        for model_file in model_files:
            assert os.path.exists(model_file), f"The model has not been converted: {model_file} does not exit."
            interpreter = tf.lite.Interpreter(model_path=model_file)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Check that it has the right number of input/output
            assert len(
                input_details) == input_len, f"Expected {input_len} input{'s' if input_len > 1 else ''}, but got: {len(input_details)}"
            assert len(
                output_details) == output_len, f"Expected {output_len} output{'s' if output_len > 1 else ''}, but got: {len(output_details)}"

    def test_model(self):
        # 1 input, 1 output
        # no NMS, no normalization
        self.aux_test(FLOAT32, include_norm=False, include_nms=False, include_threshold=False, input_len=1,
                      output_len=1)

    def test_model_quantized(self):
        # 1 input, 1 output, quantized
        # Fully quantized model int8
        # no NMS, no normalization
        self.aux_test(FULLINT8, include_norm=False, include_nms=False, include_threshold=False, input_len=1,
                      output_len=1)

    def test_model_nms(self):
        # 1 inputs, 4 outputs
        # NMS (threshold included) + normalization
        self.aux_test(FLOAT16, include_norm=False, include_nms=True, include_threshold=True, input_len=1, output_len=4)

    def test_model_nms_int8(self):
        # 3 input, 4 output
        # NMS + normalization
        self.aux_test(INT8, include_norm=True, include_nms=True, include_threshold=False, input_len=3, output_len=4)


if __name__ == '__main__':
    unittest.main()
