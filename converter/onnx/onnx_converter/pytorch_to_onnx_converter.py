from pathlib import Path

import onnx
import onnxruntime
import torch
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16

from helpers.constants import OUTPUT_DIR, DEFAULT_ONNX_NAME, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, ONNX_SUFFIX, END_COLOR, BLUE, GREEN, RED, FLOAT32, INT8, FLOAT16, FLOAT32_SUFFIX, \
    FLOAT16_SUFFIX, INT8_SUFFIX, OPSET, BATCH_SIZE, NB_CHANNEL, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, IMAGE_NAME, DEFAULT_MAX_NUMBER_DETECTION, \
    PREDICTIONS_NAME
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from pytorch_utils.pytorch_nms import YoloNMS


class PytorchToONNXConverter:

    def __init__(self, model_input_path,
                 model_output_directory=OUTPUT_DIR,
                 model_output_name=DEFAULT_ONNX_NAME,
                 input_resolution=DEFAULT_INPUT_RESOLUTION,
                 include_nms=True,
                 include_normalization=True,
                 quantization_types=None,
                 iou_threshold=DEFAULT_IOU_THRESHOLD,
                 conf_threshold=DEFAULT_CONF_THRESHOLD,
                 max_det=DEFAULT_MAX_NUMBER_DETECTION):

        self.model_input_path = Path(model_input_path)
        self.model_output_directory_path = Path(model_output_directory)
        self.model_output_name = model_output_name

        if quantization_types is None:
            quantization_types = [DEFAULT_QUANTIZATION_TYPE]
        self.quantization_types = quantization_types

        self.input_resolution = input_resolution
        self.include_nms = include_nms
        self.include_normalization = include_normalization

        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_det = max_det

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        '''
        Converts a PyTorch model to a ONNX model
        '''
        # Load the model
        print(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.__check_input_create_output()
        self.__init_pytorch_model()

        # Maybe add NMS layer and normalization
        self.model = YoloNMS(self.model, max_det=self.max_det, iou_thres=self.iou_threshold,
                             conf_thres=self.conf_threshold, normalized=self.include_normalization,
                             nmsed=self.include_nms)

        file_name = self.__get_converted_model_name()
        temp_converted_model_path = self.model_output_directory_path / file_name

        print(f'{BLUE}Converting model to ONNX with onnx {onnx.__version__}...{END_COLOR}')
        try:
            # Create a first version
            if self.include_nms:
                torch.onnx.export(self.model,
                                  self.sample_input,
                                  temp_converted_model_path, verbose=False, opset_version=OPSET,
                                  training=torch.onnx.TrainingMode.EVAL,
                                  input_names=[IMAGE_NAME],
                                  output_names=[BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME])
            else:
                torch.onnx.export(self.model,
                                  self.sample_input,
                                  temp_converted_model_path, verbose=False, opset_version=OPSET,
                                  training=torch.onnx.TrainingMode.EVAL,
                                  input_names=[IMAGE_NAME],
                                  output_names=[PREDICTIONS_NAME])

            # Checks
            model_onnx = onnx.load(temp_converted_model_path)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            onnxruntime.InferenceSession(str(temp_converted_model_path))  # check inference can be initialized
            print(f"{GREEN}ONNX conversion success")
        except Exception as e:
            raise Exception(f"{RED}ONNX conversion failure:{END_COLOR} {e}")

        for quantization_type in self.quantization_types:
            # Quantization
            print(f'{BLUE}Exporting model to ONNX {quantization_type}...{END_COLOR}')
            try:
                model_temp = onnx.load(temp_converted_model_path)
                if quantization_type == FLOAT32:
                    suffix = FLOAT32_SUFFIX
                elif quantization_type == FLOAT16:
                    suffix = FLOAT16_SUFFIX
                    model_temp = convert_float_to_float16(model_temp)
                elif quantization_type == INT8:
                    suffix = INT8_SUFFIX
                    quantize_dynamic(temp_converted_model_path, temp_converted_model_path, weight_type=QuantType.QUInt8)
                    model_temp = onnx.load(temp_converted_model_path)
                else:
                    raise ValueError(f"The quantization type '{quantization_type}' is not supported.")

                converted_model_path = self.model_output_directory_path / Path(
                    file_name.replace(ONNX_SUFFIX, suffix + ONNX_SUFFIX))
                # Add metadata (labels)
                self.__add_metadata(model_temp, self.model.names, quantization_type)

                onnx.save(model_temp, converted_model_path)
                print(f'{GREEN}ONNX export success:{END_COLOR} saved as {converted_model_path}')
            except Exception as e:
                temp_converted_model_path.unlink()
                raise Exception(f'{RED}ONNX export failure:{END_COLOR} {e}')

        temp_converted_model_path.unlink()

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            print(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_directory_path.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        self.model = PyTorchModelLoader(self.model_input_path, self.input_resolution).load(fuse=True)
        self.sample_input = torch.zeros((BATCH_SIZE, NB_CHANNEL, self.input_resolution, self.input_resolution))

    def __add_metadata(self, model, labels, quantization_type):
        # Add metadata
        meta = model.metadata_props.add()
        meta.key = 'labels'
        meta.value = ','.join(labels)

        meta = model.metadata_props.add()
        meta.key = 'quantized'
        meta.value = quantization_type

        meta = model.metadata_props.add()
        meta.key = 'do_normalize'
        meta.value = str(self.include_normalization)

        meta = model.metadata_props.add()
        meta.key = 'do_nms'
        meta.value = str(self.include_nms)

    def __get_converted_model_name(self):
        basename = self.model_output_name

        if not self.include_normalization:
            basename += '_no-norm'
        if not self.include_nms:
            basename += '_no-nms'

        basename += f'_{self.input_resolution}'
        return basename + ONNX_SUFFIX
