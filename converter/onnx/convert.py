from argparse import ArgumentParser

from helpers.constants import DEFAULT_MODEL_OUTPUT_DIR, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, DEFAULT_ONNX_NAME, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    DEFAULT_MAX_NUMBER_DETECTION
from onnx_converter.pytorch_to_onnx_converter import PytorchToONNXConverter
from helpers.parameters import ModelParameters, ConversionParameters

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, dest="model_input_path", required=True,
                        help=f"The path to yolov5 model.")
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
    parser.add_argument('--no-nms', action='store_true', help='If set, NMS is not added at the end of the model.')
    parser.add_argument('--no-normalization', action='store_true',
                        help='If set, normalization is not added at the beginning of the model.')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, overwrites already converted model if it exists.')

    opt = parser.parse_args()

    model_parameters = ModelParameters(input_resolution=opt.input_resolution, include_nms=not opt.no_nms,
                                       include_normalization=not opt.no_normalization, max_det=opt.max_det)

    conversion_parameters = ConversionParameters(quantization_types=opt.quantization_types,
                                                 iou_threshold=opt.iou_threshold,
                                                 conf_threshold=opt.conf_threshold)

    converter = PytorchToONNXConverter(model_input_path=opt.model_input_path,
                                       model_parameters=model_parameters,
                                       conversion_parameters=conversion_parameters,
                                       overwrite=opt.overwrite)

    converter.convert()


if __name__ == '__main__':
    main()
