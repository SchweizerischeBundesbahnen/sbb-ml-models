import torch
from torch import nn

from helpers.constants import DEFAULT_INPUT_RESOLUTION, BATCH_SIZE, NB_CHANNEL, SEGMENTATION, DETECTION
from models.experimental import attempt_load
from models.yolo import Segment, Detect


class PyTorchModelLoader:

    def __init__(self, model_input_path, input_resolution=DEFAULT_INPUT_RESOLUTION):
        self.model_input_path = model_input_path
        self.input_resolution = input_resolution

    def __load_model(self, fuse):
        try:
            self.torch_model = attempt_load(self.model_input_path, device='cpu', inplace=True, fuse=fuse)
        except:
            self.torch_model = attempt_load(self.model_input_path)
        if isinstance(self.torch_model.names, dict):
            self.torch_model.names = list(self.torch_model.names.values())

    def __dry_run(self):
        self.torch_model.eval()  # Will return predictions, model outputs
        sample_input = torch.zeros((BATCH_SIZE, NB_CHANNEL, self.input_resolution, self.input_resolution))
        self.torch_model(sample_input)

    def __load_attributes(self):
        self.model_type = SEGMENTATION if isinstance(self.torch_model.model[-1], Segment) else DETECTION
        self.class_labels = self.torch_model.names
        self.strides = self.torch_model.stride
        self.feature_map_dimensions = [self.input_resolution // int(stride) for stride in self.strides]
        self.anchors = self.torch_model.model[-1].anchors
        self.input_shape = (BATCH_SIZE, NB_CHANNEL, self.input_resolution, self.input_resolution)
        self.number_of_classes = len(self.class_labels)
        if isinstance(self.torch_model.model[-1], Segment):
            self.model_type = SEGMENTATION
            self.number_of_masks = self.torch_model.model[-1].nm
        elif isinstance(self.torch_model.model[-1], Detect):
            self.model_type = DETECTION
            self.number_of_masks = None
        else:
            raise ValueError(
                f"Yolo model should end with either 'Segment' or 'Detect' not {type(self.torch_model.model[-1])}")

    def load(self, fuse=True):
        self.__load_model(fuse)
        self.__dry_run()
        self.__load_attributes()
        return self