from argparse import ArgumentParser

from helpers.constants import DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, DEFAULT_ONNX_NAME, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    DEFAULT_MAX_NUMBER_DETECTION
from onnx_converter.pytorch_to_onnx_converter import PytorchToONNXConverter


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, dest="model_input_path", required=True,
                        help=f"The path to yolov5 model.")
    parser.add_argument('--out', type=str, dest="model_output_directory", default=DEFAULT_MODEL_OUTPUT_DIR,
                        help=f"The path to the directory in which to save the converted model. Default: {DEFAULT_MODEL_OUTPUT_DIR}")
    parser.add_argument('--output-name', type=str, dest="model_output_name",
                        default=DEFAULT_ONNX_NAME, help=f'The model output name. Default: {DEFAULT_ONNX_NAME}')
    parser.add_argument('--input-resolution', type=int, dest="input_resolution", default=DEFAULT_INPUT_RESOLUTION,
                        help=f'The resolution of the input images, e.g. {DEFAULT_INPUT_RESOLUTION} means input resolution is {DEFAULT_INPUT_RESOLUTION}x{DEFAULT_INPUT_RESOLUTION}. Default: {DEFAULT_INPUT_RESOLUTION}')  # height, width
    parser.add_argument('--quantize-model', nargs='+', dest="quantization_types", default=[DEFAULT_QUANTIZATION_TYPE],
                        help=f"Quantization: 'int8', 'float16' or 'float32' for no quantization. Default: [{DEFAULT_QUANTIZATION_TYPE}]")
    parser.add_argument('--max-det', type=int, default=DEFAULT_MAX_NUMBER_DETECTION,
                        help=f'The maximum number of detections. Default: {DEFAULT_MAX_NUMBER_DETECTION}.')
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f"The IoU threshold, if set in the model. Default: {DEFAULT_IOU_THRESHOLD}")
    parser.add_argument('--conf-threshold', type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"The confidence threshold, if set in the model. Default: {DEFAULT_CONF_THRESHOLD}")
    opt = parser.parse_args()

    converter = PytorchToONNXConverter(model_input_path=opt.model_input_path,
                                       model_output_directory=opt.model_output_directory,
                                       model_output_name=opt.model_output_name,
                                       input_resolution=opt.input_resolution,
                                       quantization_types=opt.quantization_types,
                                       max_det=opt.max_det,
                                       iou_threshold=opt.iou_threshold,
                                       conf_threshold=opt.conf_threshold)

    converter.convert()


if __name__ == '__main__':
    main()
