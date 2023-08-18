import logging
import ast
import numpy as np
import onnx
import onnxruntime
import torch

from helpers.constants import BATCH_SIZE, FLOAT16, IMAGE_NAME, SCORES_NAME, CLASSES_NAME, BLUE, END_COLOR, \
    BOUNDINGBOX_NAME, PREDICTIONS_NAME, DEFAULT_MAX_NUMBER_DETECTION, DEFAULT_INPUT_RESOLUTION, SEGMENTATION
from tf_model.tf_nms import NMS
from python_model.inference_model_abs import InferenceModelAbs
from pytorch_utils.pytorch_nms import YoloNMS
from helpers.coordinates import pt_yxyx2xyxy_yolo
from utils.segment.general import process_mask

class ONNXModel(InferenceModelAbs):
    """ Class to load a ONNX model and run inference

        Attributes
        ----------
        model_path: str
            The path to the ONNX model
        """
    def __init__(self, model_path):
        logging.info(f"{BLUE}Initializing ONNX model...{END_COLOR}")
        self.model = onnx.load(model_path)
        self.session = onnxruntime.InferenceSession(model_path)

        # Load metadata
        self.__load_metadata()

    def predict(self, img, iou_threshold, conf_threshold):
        if self.quantization_type == FLOAT16:
            img = np.array(img, dtype=np.float16)

        if self.do_nms:
            outputs = self.session.run(None, {IMAGE_NAME: np.array(img)})
            predictions = torch.from_numpy(outputs[0])

            nms = YoloNMS(None, iou_thres=iou_threshold, conf_thres=conf_threshold, class_labels=self.labels)
            yxyx, classes, scores, masks = nms.nms_yolov5(predictions, self.model_orig)
            nb_detected = classes.shape[1]

            if self.model_type == SEGMENTATION:
                protos = torch.from_numpy(outputs[1])
                xyxy = pt_yxyx2xyxy_yolo(yxyx[0][:nb_detected])
                masks = process_mask(protos[0], masks[0], xyxy, self.img_size, upsample=True)  # HWC
            else:
                masks = None

            yxyx /= self.img_size[0]

            return yxyx, classes, scores, masks, torch.Tensor([nb_detected])
        else:
            # NMS is included in the model
            outputs = self.session.run(None, {IMAGE_NAME: np.array(img)})
            yxyx = outputs[0]
            classes = outputs[1]
            scores = outputs[2]
            nb_detected = classes.shape[1]
            if self.model_type == SEGMENTATION:
                masks = torch.from_numpy(outputs[3])
                protos = torch.from_numpy(outputs[4])
                xyxy = pt_yxyx2xyxy_yolo(yxyx[0][:nb_detected])
                masks = process_mask(protos[0], masks[0], xyxy, self.img_size, upsample=True)  # HWC
            else:
                masks = None


            if self.quantization_type == FLOAT16:
                yxyx = np.array(yxyx, dtype=np.float32)

            yxyx /= self.img_size[0]

            return yxyx, classes, scores, masks, torch.Tensor([nb_detected])

    def __load_metadata(self):
        self.img_size = (DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION)
        self.input_dict = {x.name : [d.dim_value for d in x.type.tensor_type.shape.dim] for x in self.model.graph.input}
        self.output_dict = {x.name : [d.dim_value for d in x.type.tensor_type.shape.dim] for x in self.model.graph.output}

        metadata = self.model.metadata_props
        for m in metadata:
            if m.key == 'names':
                self.labels = ast.literal_eval(m.value)
            if m.key == 'quantization_type':
                self.quantization_type = m.value
            if m.key == 'normalized':
                self.do_normalize = False if m.value == 'True' else True
            if m.key == 'nms':
                self.do_nms = False if m.value == 'True' else True
            if m.key == 'task':
                self.model_type = m.value
            if m.key == 'orig':
                self.model_orig = m.value

        self.batch_size = BATCH_SIZE
        self.pil_image = False
        self.channel_first = True

        # Get input size
        input_shape = self.session.get_inputs()[0].shape
        self.batch_size = int(input_shape[0])
        self.img_size = (int(input_shape[2]), int(input_shape[3]))
