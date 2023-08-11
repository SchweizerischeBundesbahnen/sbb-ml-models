"""Converts a Pytorch Yolo model to a TFLite model.

Usage
----------
```shell
convert.py [-h] --model MODEL_INPUT_PATH [--input-resolution INPUT_RESOLUTION]
                  [--quantize-model QUANTIZATION_TYPES [QUANTIZATION_TYPES ...]] [--max-det MAX_DET] [--no-nms] [--no-normalization]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL_INPUT_PATH
                        The path to yolov5 model.
  --input-resolution INPUT_RESOLUTION
                        The resolution of the input images, e.g. 640 means input resolution is 640x640. Default: 640
  --quantize-model QUANTIZATION_TYPES [QUANTIZATION_TYPES ...]
                        Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [float32]
  --max-det MAX_DET     The maximum number of detections if NMS is used. Default: 20.
  --no-nms              If set, NMS is not added at the end of the model.
  --no-normalization    If set, normalization is not added at the beginning of the model.
  ```
"""

import argparse

from helpers.constants import DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, DEFAULT_MAX_NUMBER_DETECTION

from tf_converter.pytorch_to_tf_converter import PytorchToTFConverter
from tf_utils.parameters import ModelParameters, ConversionParameters


def main():
    """Converts a Pytorch Yolo model to a TFLite model."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, dest="model_input_path", required=True,
                        help=f"The path to yolov5 model.")
    parser.add_argument('--input-resolution', type=int, dest="input_resolution", default=DEFAULT_INPUT_RESOLUTION,
                        help=f'The resolution of the input images, e.g. {DEFAULT_INPUT_RESOLUTION} means input resolution is {DEFAULT_INPUT_RESOLUTION}x{DEFAULT_INPUT_RESOLUTION}. Default: {DEFAULT_INPUT_RESOLUTION}')  # height, width
    parser.add_argument('--quantize-model', nargs='+', dest='quantization_types', default=[DEFAULT_QUANTIZATION_TYPE],
                        help=f"Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [{DEFAULT_QUANTIZATION_TYPE}]")
    parser.add_argument('--max-det', type=int, default=DEFAULT_MAX_NUMBER_DETECTION,
                        help=f'The maximum number of detections. Default: {DEFAULT_MAX_NUMBER_DETECTION}.')
    parser.add_argument('--no-nms', action='store_true', help='If set, NMS is not added at the end of the model.')
    parser.add_argument('--no-normalization', action='store_true',
                        help='If set, normalization is not added at the beginning of the model.')

    opt = parser.parse_args()

    model_parameters = ModelParameters(input_resolution=opt.input_resolution, include_nms=not opt.no_nms,
                                       include_normalization=not opt.no_normalization, max_det=opt.max_det)

    conversion_parameters = ConversionParameters(quantization_types=opt.quantization_types)

    converter = PytorchToTFConverter(model_input_path=opt.model_input_path,
                                     model_parameters=model_parameters,
                                     conversion_parameters=conversion_parameters)

    converter.convert()


if __name__ == "__main__":
    main()
