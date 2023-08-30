from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from helpers.constants import ONNX, \
    INT8, FLOAT16, OPSET, BATCH_SIZE, NB_CHANNEL, BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, OUTPUT_DIR, \
    PREDICTIONS_NAME, IMAGE_NAME, SEGMENTATION, MASKS_NAME, PROTOS_NAME
from helpers.converter import Converter
from helpers.parameters import ModelParameters, ConversionParameters
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16
from pytorch_utils.pytorch_nms import YoloNMS


class PytorchToONNXConverter(Converter):
    """ Class that converts a Pytorch model to a ONNX model

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

    def __init__(self, model_input_path: str,
                 model_output_path: str = None,
                 model_parameters: ModelParameters = ModelParameters(),
                 conversion_parameters: ConversionParameters = ConversionParameters(),
                 overwrite: bool = False):

        super().__init__(model_input_path=model_input_path,
                         model_output_path=model_output_path,
                         model_parameters=model_parameters,
                         conversion_parameters=conversion_parameters,
                         overwrite=overwrite, destination=ONNX)
        self.sample_input = torch.zeros(
            (BATCH_SIZE, NB_CHANNEL, self.model_parameters.input_resolution, self.model_parameters.input_resolution))

    def _convert(self):
        # Maybe add NMS layer and normalization
        self.model = YoloNMS(self.pt_model, max_det=self.model_parameters.max_det,
                             iou_thres=self.conversion_parameters.iou_threshold,
                             conf_thres=self.conversion_parameters.conf_threshold,
                             normalized=self.model_parameters.include_normalization,
                             nmsed=self.model_parameters.include_nms)

        self.temp_converted_model_path = Path(OUTPUT_DIR) / "temp.onnx"
        # Create a first version
        self.inputs = [IMAGE_NAME]
        self.outputs = [PREDICTIONS_NAME]
        if self.model_parameters.model_type == SEGMENTATION:
            self.outputs = [PREDICTIONS_NAME, PROTOS_NAME]
        if self.model_parameters.include_nms:
            if self.model_parameters.model_type == SEGMENTATION:
                self.outputs = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, MASKS_NAME]
            else:
                self.outputs = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME]

        torch.onnx.export(self.model,
                          self.sample_input,
                          self.temp_converted_model_path, verbose=False, opset_version=OPSET,
                          training=torch.onnx.TrainingMode.EVAL,
                          input_names=self.inputs,
                          output_names=self.outputs)

        # Checks
        model_onnx = onnx.load(self.temp_converted_model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        return model_onnx

    def _dry_run(self, model):
        session = onnxruntime.InferenceSession(
            str(self.temp_converted_model_path))  # check inference can be initialized
        session.run(None, {IMAGE_NAME: np.array(self.sample_input)})

    def _export_with_quantization(self, model, quantization_type):
        converted_model = model
        if quantization_type == FLOAT16:
            converted_model = convert_float_to_float16(converted_model)
        elif quantization_type == INT8:
            quantize_dynamic(self.temp_converted_model_path, self.temp_converted_model_path,
                             weight_type=QuantType.QUInt8)
            converted_model = onnx.load(self.temp_converted_model_path)
        return converted_model

    def _set_metadata(self, model, quantization_type):
        metadata = self.model_parameters.metadata
        metadata.update({'input_names': self.inputs,
                         'output_names': self.outputs,
                         'quantization_type': quantization_type})

        for k, v in metadata.items():
            meta = model.metadata_props.add()
            meta.key, meta.value = k, str(v)

    def _save_model(self, converted_model, converted_model_path):
        onnx.save(converted_model, converted_model_path)

    def _on_end(self):
        Path(self.temp_converted_model_path).unlink()
