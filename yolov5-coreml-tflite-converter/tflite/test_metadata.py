import unittest

from constants import IMAGE_NAME, NORMALIZED_SUFFIX, QUANTIZED_SUFFIX, IOU_NAME, CONF_NAME, PREDICTIONS_NAME, \
    BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, NUMBER_NAME, FLOAT32, FLOAT16, INT8, FULLINT8
from tf_metadata.metadata_writer import MetadataWriter
from tf_utils.parameters import ModelParameters


class MetadataTest(unittest.TestCase):
    '''
    Checks that the metadata is written correctly given different model configurations
    '''

    @staticmethod
    def aux_test(inputs, outputs, include_normalization, include_nms, include_threshold, quantized, input_len,
                 input_names,
                 output_len, output_names):
        model_parameters = ModelParameters(include_nms=include_nms, include_normalization=include_normalization,
                                           include_threshold=include_threshold, img_size=[640, 640])
        metadata_writer = MetadataWriter(inputs, outputs, model_parameters=model_parameters,
                                         labels=['Some', 'labels'], max_det=20,
                                         quantized=quantized, tflite_path=None)
        try:
            metadata_writer.write()
        except ValueError as e:
            # Error is thrown because no tflite_file is no specified
            assert str(
                e) == "ERROR: Expected binary or unicode string, got None", f"Unexpected error: {e}"

        # Check the input metadata: correct length, correct names
        assert len(
            metadata_writer.input_meta) == input_len, f"The metadata for the input should have length {input_len} but has length {len(metadata_writer.input_meta)}."
        for i, input_name in enumerate(input_names):
            m_input_name = metadata_writer.input_meta[i].name
            assert m_input_name == input_name, f"Expected input: {input_name}, but got: {m_input_name}."

        # Check the output metadata: correct length, correct names
        assert len(
            metadata_writer.output_meta) == output_len, f"The metadata for the output should have length {output_len} but has length {len(metadata_writer.output_meta)}."
        for i, output_name in enumerate(output_names):
            m_output_name = metadata_writer.output_meta[i].name
            assert m_output_name == output_name, f"Expected output: {output_name}, but got: {m_output_name}."

    def test_metadata_writer(self):
        # 1 input normalized, 1 output
        # No NMS, no normalization
        for quant_type, input_name in zip([FLOAT32, FLOAT16, INT8, FULLINT8],
                                          [[IMAGE_NAME + NORMALIZED_SUFFIX], [IMAGE_NAME + NORMALIZED_SUFFIX],
                                           [IMAGE_NAME + NORMALIZED_SUFFIX],
                                           [IMAGE_NAME + QUANTIZED_SUFFIX + NORMALIZED_SUFFIX]]):
            self.aux_test([IMAGE_NAME], [PREDICTIONS_NAME], include_normalization=False, include_nms=False,
                          include_threshold=False, quantized=quant_type, input_len=1, input_names=input_name,
                          output_len=1,
                          output_names=[PREDICTIONS_NAME])

    def test_metadata_writer_norm(self):
        # 1 input, 1 output
        # No NMS, but normalization included
        for quant_type, input_name in zip([FLOAT32, FLOAT16, INT8, FULLINT8],
                                          [[IMAGE_NAME], [IMAGE_NAME], [IMAGE_NAME],
                                           [IMAGE_NAME + QUANTIZED_SUFFIX]]):
            self.aux_test([IMAGE_NAME], [PREDICTIONS_NAME], include_normalization=True, include_nms=False,
                          include_threshold=False, quantized=quant_type, input_len=1, input_names=input_name,
                          output_len=1,
                          output_names=[PREDICTIONS_NAME])

    def test_metadata_writer_nms(self):
        # 1 input, 4 outputs
        # NMS included (with thresholds hard coded), but no normalization
        for quant_type in [FLOAT32, FLOAT16, INT8]:
            self.aux_test([IMAGE_NAME], [SCORES_NAME, NUMBER_NAME, BOUNDINGBOX_NAME, CLASSES_NAME],
                          include_normalization=False,
                          include_nms=True, include_threshold=True, quantized=quant_type, input_len=1,
                          input_names=[IMAGE_NAME + NORMALIZED_SUFFIX],
                          output_len=4, output_names=[SCORES_NAME, NUMBER_NAME, BOUNDINGBOX_NAME, CLASSES_NAME])

    def test_metadata_writer_nms_with_inputs(self):
        # 3 inputs, 4 outputs
        # NMS included (with thresholds as inputs), normalization too
        for quant_type in [FLOAT32, FLOAT16, INT8]:
            self.aux_test([IOU_NAME, IMAGE_NAME, CONF_NAME], [SCORES_NAME, NUMBER_NAME, BOUNDINGBOX_NAME, CLASSES_NAME],
                          include_normalization=True, include_nms=True, include_threshold=False, quantized=quant_type,
                          input_len=3, input_names=[IOU_NAME, IMAGE_NAME, CONF_NAME], output_len=4,
                          output_names=[SCORES_NAME, NUMBER_NAME, BOUNDINGBOX_NAME, CLASSES_NAME])


if __name__ == '__main__':
    unittest.main()
