"""Converts a Pytorch Yolo model to a TFLite model.

Usage
----------
```shell
convert.py [-h] --model MODEL_INPUT_PATH [--model-type MODEL_TYPE] [--out MODEL_OUTPUT_DIRECTORY]
                  [--output-name MODEL_OUTPUT_NAME] [--input-resolution INPUT_RESOLUTION [INPUT_RESOLUTION ...]]
                  [--quantize-model QUANTIZATION_TYPES [QUANTIZATION_TYPES ...]] [--no-nms] [--no-normalization]
                  [--max-det MAX_DET] [--nms-type NMS_TYPE]
```

optional arguments:
```shell
  -h, --help            show this help message and exit
  --model MODEL_INPUT_PATH
                        The path to yolov5 model.
  --out MODEL_OUTPUT_DIRECTORY
                        The path to the directory in which to save the converted model. Default: data/output/converted_models
  --output-name MODEL_OUTPUT_NAME
                        The model output name. Default: yolov5-TFLite
  --input-resolution INPUT_RESOLUTION [INPUT_RESOLUTION ...]
                        The resolution of the input images, e.g. 640 means input resolution is 640x640. Default: 640
  --quantize-model QUANTIZATION_TYPES [QUANTIZATION_TYPES ...]
                        Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [float32]
  --no-nms              If set, the converted model does not include the postprocessing (NMS)
  --no-normalization    If set, the converted model does not include the preprocessing (normalization)
  --max-det MAX_DET     The maximum number of detections. Default: 20.
  --nms-type NMS_TYPE   NMS algorithm used: one of 'simple', 'padded', 'combined'.
  ```
"""

import argparse

from helpers.constants import DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_TFLITE_NAME, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, DEFAULT_MAX_NUMBER_DETECTION, SIMPLE, PADDED, \
    COMBINED
from tf_converter.pytorch_to_tf_converter import PytorchToTFConverter
from tf_utils.parameters import ModelParameters, ConversionParameters, PostprocessingParameters


def main():
    """Converts a Pytorch Yolo model to a TFLite model."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, dest="model_input_path", required=True,
                        help=f"The path to yolov5 model.")
    parser.add_argument('--out', type=str, dest="model_output_directory", default=DEFAULT_MODEL_OUTPUT_DIR,
                        help=f"The path to the directory in which to save the converted model. Default: {DEFAULT_MODEL_OUTPUT_DIR}")
    parser.add_argument('--output-name', type=str, dest="model_output_name",
                        default=DEFAULT_TFLITE_NAME, help=f'The model output name. Default: {DEFAULT_TFLITE_NAME}')
    parser.add_argument('--input-resolution', type=int, dest="input_resolution", default=DEFAULT_INPUT_RESOLUTION,
                        help=f'The resolution of the input images, e.g. {DEFAULT_INPUT_RESOLUTION} means input resolution is {DEFAULT_INPUT_RESOLUTION}x{DEFAULT_INPUT_RESOLUTION}. Default: {DEFAULT_INPUT_RESOLUTION}')  # height, width
    parser.add_argument('--quantize-model', nargs='+', dest='quantization_types', default=[DEFAULT_QUANTIZATION_TYPE],
                        help=f"Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [{DEFAULT_QUANTIZATION_TYPE}]")

    # Optional parameters
    parser.add_argument('--no-nms', action='store_true',
                        help='If set, the converted model does not include the postprocessing (NMS)')
    parser.add_argument('--no-normalization', action='store_true',
                        help='If set, the converted model does not include the preprocessing (normalization)')

    parser.add_argument('--max-det', type=int, default=DEFAULT_MAX_NUMBER_DETECTION,
                        help=f'The maximum number of detections. Default: {DEFAULT_MAX_NUMBER_DETECTION}.')
    parser.add_argument('--nms-type', default=PADDED,
                        help=f"NMS algorithm used: one of '{SIMPLE}', '{PADDED}', '{COMBINED}'.")

    opt = parser.parse_args()

    model_parameters = ModelParameters(input_resolution=opt.input_resolution,
                                       include_nms=not opt.no_nms,
                                       include_normalization=not opt.no_normalization)

    conversion_parameters = ConversionParameters(quantization_types=opt.quantization_types)
    postprocessing_parameters = PostprocessingParameters(max_det=opt.max_det, nms_type=opt.nms_type)

    converter = PytorchToTFConverter(model_input_path=opt.model_input_path,
                                     model_output_directory=opt.model_output_directory,
                                     model_output_name=opt.model_output_name,
                                     model_parameters=model_parameters,
                                     conversion_parameters=conversion_parameters,
                                     postprocessing_parameters=postprocessing_parameters)

    converter.convert()


if __name__ == "__main__":
    main()
