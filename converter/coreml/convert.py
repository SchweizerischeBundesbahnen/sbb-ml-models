# Copyright (C) 2021 DB Systel GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########
# Copyright (C) 2021 SBB
#
# Modification statement in accordance with the Apache 2.0 license:
# This code was modified by Ferdinand Niedermann for SBB in 2021.
# The modifications are NOT published under the Apache 2.0 license.
########
"""Converts a Pytorch Yolo model to a CoreML model.

Usage
----------
```shell
convert.py [-h] --model MODEL_INPUT_PATH [--out MODEL_OUTPUT_DIRECTORY]
                  [--output-name MODEL_OUTPUT_NAME] [--input-resolution INPUT_RESOLUTION [INPUT_RESOLUTION ...]]
                  [--quantize-model QUANTIZATION_TYPES [QUANTIZATION_TYPES ...]] [--max-det MAX_DET]
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
  --max-det MAX_DET     The maximum number of detections (for segmentation). Default: 20.
  ```
"""

from argparse import ArgumentParser

from coreml_converter.pytorch_to_coreml_converter import PytorchToCoreMLConverter
from helpers.constants import DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_COREML_NAME, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, DEFAULT_MAX_NUMBER_DETECTION


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, dest="model_input_path", required=True,
                        help=f"The path to yolov5 model.")
    parser.add_argument('--out', type=str, dest="model_output_directory", default=DEFAULT_MODEL_OUTPUT_DIR,
                        help=f"The path to the directory in which to save the converted model. Default: {DEFAULT_MODEL_OUTPUT_DIR}")
    parser.add_argument('--output-name', type=str, dest="model_output_name",
                        default=DEFAULT_COREML_NAME, help=f'The model output name. Default: {DEFAULT_COREML_NAME}')
    parser.add_argument('--input-resolution', type=int, dest="input_resolution", default=DEFAULT_INPUT_RESOLUTION,
                        help=f'The resolution of the input images, e.g. {DEFAULT_INPUT_RESOLUTION} means input resolution is {DEFAULT_INPUT_RESOLUTION}x{DEFAULT_INPUT_RESOLUTION}. Default: {DEFAULT_INPUT_RESOLUTION}')  # height, width
    parser.add_argument('--quantize-model', nargs='+', dest="quantization_types", default=[DEFAULT_QUANTIZATION_TYPE],
                        help=f"Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [{DEFAULT_QUANTIZATION_TYPE}]")
    parser.add_argument('--max-det', type=int, default=DEFAULT_MAX_NUMBER_DETECTION,
                        help=f'The maximum number of detections. Default: {DEFAULT_MAX_NUMBER_DETECTION}.')
    opt = parser.parse_args()

    converter = PytorchToCoreMLConverter(opt.model_input_path, opt.model_output_directory, opt.model_output_name,
                                         opt.quantization_types, opt.input_resolution, opt.max_det)
    converter.convert()


if __name__ == '__main__':
    main()
