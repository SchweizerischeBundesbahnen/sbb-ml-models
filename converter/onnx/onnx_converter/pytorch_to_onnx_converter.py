from pathlib import Path
import logging
import onnx
import onnxruntime
import torch
from helpers.constants import OUTPUT_DIR, DEFAULT_ONNX_NAME, DEFAULT_INPUT_RESOLUTION, \
    DEFAULT_QUANTIZATION_TYPE, ONNX_SUFFIX, END_COLOR, BLUE, GREEN, RED, FLOAT32, INT8, FLOAT16, FLOAT32_SUFFIX, \
    FLOAT16_SUFFIX, INT8_SUFFIX, OPSET, BATCH_SIZE, NB_CHANNEL, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, DEFAULT_MAX_NUMBER_DETECTION, \
    PREDICTIONS_NAME, IMAGE_NAME, SEGMENTATION, MASKS_NAME, PT_SUFFIX, DETECTION, PROTOS_NAME
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from pytorch_utils.pytorch_nms import YoloNMS
from tf_utils.parameters import ModelParameters


class PytorchToONNXConverter:
    """ Class that converts a Pytorch model to a ONNX model

        Attributes
        ----------
        model_input_path: String
            The path to the Pytorch model

        model_output_path: String
            The path to the converted model

        quantization_types: List[str]
            The quantization types to use for the conversion

        input_resolution: int
            The input resolution of the image

        include_nms: bool
            Whether the converted model contains NMS

        include_normalization: bool
            Whether the converter model contains normalization

        iou_threshold: float
            The IoU threshold if the model contains NMS

        conf_threshold: float
            The confidence threshold if the model contains NMS

        max_det: int
            The maximum number of detections that can be made by the CoreML model
        """

    def __init__(self, model_input_path,
                 model_output_path=None,
                 model_parameters=ModelParameters(),
                 quantization_types=None,
                 iou_threshold=DEFAULT_IOU_THRESHOLD,
                 conf_threshold=DEFAULT_CONF_THRESHOLD,
                 max_det=DEFAULT_MAX_NUMBER_DETECTION):

        self.model_input_path = Path(model_input_path)
        self.model_output_path = Path(model_output_path) if model_output_path else Path(
            model_input_path.replace(PT_SUFFIX, ''))

        if quantization_types is None:
            quantization_types = [DEFAULT_QUANTIZATION_TYPE]
        self.quantization_types = quantization_types

        self.model_parameters = model_parameters

        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_det = max_det

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        '''
        Converts a PyTorch model to a ONNX model
        '''

        # Maybe add NMS layer and normalization
        self.model = YoloNMS(self.pt_model, max_det=self.max_det, iou_thres=self.iou_threshold,
                             conf_thres=self.conf_threshold, normalized=self.model_parameters.include_normalization,
                             nmsed=self.model_parameters.include_nms)

        temp_converted_model_path = self.__get_converted_model_name()

        print(f'{BLUE}Converting model to ONNX with onnx {onnx.__version__}...{END_COLOR}')
        try:
            # Create a first version
            inputs = [IMAGE_NAME]
            outputs = [PREDICTIONS_NAME]
            if self.model_parameters.model_type == SEGMENTATION:
                outputs = [PREDICTIONS_NAME, PROTOS_NAME]
            if self.model_parameters.include_nms:
                if self.model_parameters.model_type == SEGMENTATION:
                    outputs = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, MASKS_NAME]
                else:
                    outputs = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME]

            torch.onnx.export(self.model,
                              self.sample_input,
                              temp_converted_model_path, verbose=False, opset_version=OPSET,
                              training=torch.onnx.TrainingMode.EVAL,
                              input_names=inputs,
                              output_names=outputs)

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

                converted_model_path = temp_converted_model_path.replace(ONNX_SUFFIX, suffix + ONNX_SUFFIX)
                # Add metadata (labels)
                self.__add_metadata(model_temp, inputs, outputs, quantization_type)

                onnx.save(model_temp, converted_model_path)
                print(f'{GREEN}ONNX export success:{END_COLOR} saved as {converted_model_path}')
            except Exception as e:
                Path(temp_converted_model_path).unlink()
                raise Exception(f'{RED}ONNX export failure:{END_COLOR} {e}')

        Path(temp_converted_model_path).unlink()

    def __add_metadata(self, model, inputs, outputs, quantization_type):
        metadata = self.model_parameters.metadata
        metadata.update({'input_names': inputs,
                         'output_names': outputs,
                         'quantization_type': quantization_type})

        for k, v in metadata.items():
            meta = model.metadata_props.add()
            meta.key, meta.value = k, str(v)

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            print(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        self.pt_model = PyTorchModelLoader(self.model_input_path, self.model_parameters).load(fuse=True)
        self.sample_input = torch.zeros((BATCH_SIZE, NB_CHANNEL, self.model_parameters.input_resolution, self.model_parameters.input_resolution))

        logging.info(f"{BLUE}\t- The model has {self.model_parameters.nb_classes} classes{END_COLOR}")
        logging.info(
            f"{BLUE}\t- The model is for {self.model_parameters.model_type} ({self.model_parameters.model_orig}){END_COLOR}")
        logging.info(
            f"{BLUE}\t- Normalization will{'' if self.model_parameters.include_normalization else ' not'} be included in the model.{END_COLOR}")
        logging.info(
            f"{BLUE}\t- NMS will{'' if self.model_parameters.include_nms else ' not'} be included in the model.{END_COLOR}")

    def __get_converted_model_name(self):
        basename = str(self.model_output_path)

        if not self.model_parameters.include_normalization:
            basename += '_no-norm'
        if not self.model_parameters.include_nms:
            basename += '_no-nms'

        basename += f'_{self.model_parameters.input_resolution}'
        return basename + ONNX_SUFFIX
