import logging

import torch
from helpers.constants import BATCH_SIZE, DEFAULT_INPUT_RESOLUTION
from helpers.coordinates import pt_yxyx2xyxy_yolo
from pytorch_utils.pytorch_loader import PyTorchModelLoader
from pytorch_utils.pytorch_nms import YoloNMS
from utils.segment.general import process_mask
from helpers.parameters import ModelParameters
from python_model.inference_model import InferenceModel

class PyTorchModel(InferenceModel):
    """ Class to load a PyTorch model and run inference

        Attributes
        ----------
        model_path: str
            The path to the PyTorch model

        input_resolution: int
            The input resolution for the input to the model
    """

    def __init__(self, model_path, input_resolution=DEFAULT_INPUT_RESOLUTION):
        logging.info("- Initializing PyTorch model...")
        model_parameters = ModelParameters(input_resolution=input_resolution)
        self.pt_model = PyTorchModelLoader(model_path, model_parameters).load(fuse=True)
        self.__load_metadata()
        self.img_size = (input_resolution, input_resolution)
        super().__init__(model_path)

    def predict(self, img, iou_threshold, conf_threshold):
        model = YoloNMS(self.pt_model, iou_thres=iou_threshold, conf_thres=conf_threshold)

        (yxyx, classes, scores, masks), protos = model(img)

        nb_detected = classes.shape[1]

        xyxy = pt_yxyx2xyxy_yolo(yxyx[0][:nb_detected])

        if masks.shape[2] != 0:
            masks = process_mask(protos[0], masks[0], xyxy, self.img_size, upsample=True)  # HWC
        else:
            masks = None

        yxyx /= self.img_size[0]

        return yxyx, classes, scores, masks, torch.Tensor([nb_detected])

    def __load_metadata(self):
        self.do_normalize = False
        self.do_nms = True
        self.batch_size = BATCH_SIZE
        self.pil_image = False
        self.channel_first = True
        self.labels = self.pt_model.torch_model.names
        self.model_type = self.pt_model.model_parameters.model_type
        self.input_names = []
        self.output_names = []