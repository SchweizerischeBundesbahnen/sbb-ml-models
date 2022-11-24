import torch
from models.experimental import attempt_load
from torch import nn

from helpers.constants import DEFAULT_INPUT_RESOLUTION, BATCH_SIZE, NB_CHANNEL


class PyTorchModelLoader:

    def __init__(self, model_input_path, input_resolution=DEFAULT_INPUT_RESOLUTION):
        self.model_input_path = model_input_path
        self.input_resolution = input_resolution

    def __load_model(self, fuse):
        self.model = attempt_load(self.model_input_path, map_location='cpu', inplace=True, fuse=fuse)

    def __dry_run(self):
        self.model.eval()  # Will return predictions, model outputs
        sample_input = torch.zeros((BATCH_SIZE, NB_CHANNEL, self.input_resolution, self.input_resolution))
        self.model(sample_input)

    def load(self, fuse=True):
        self.__load_model(fuse)
        self.__dry_run()
        return self.model

    def load_wrapper(self):
        # Used for CoreML, contains information about strides, feature map and anchors
        self.__load_model(fuse=True)
        self.__dry_run()
        return ModelWrapper(self.model, self.input_resolution)


class ModelWrapper:
    def __init__(self, model, input_resolution):
        self.torch_model = model
        self.class_labels = model.names
        self.strides = model.stride
        self.feature_map_dimensions = [input_resolution // int(stride) for stride in self.strides]
        self.anchors = model.model[-1].anchors
        self.input_shape = (BATCH_SIZE, NB_CHANNEL, input_resolution, input_resolution)
        self.input_resolution = input_resolution

    def number_of_classes(self):
        return len(self.class_labels)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
