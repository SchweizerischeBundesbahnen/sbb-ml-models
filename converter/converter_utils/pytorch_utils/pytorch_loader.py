from datetime import datetime
from pathlib import Path

import torch
from helpers.constants import BATCH_SIZE, NB_CHANNEL, SEGMENTATION, DETECTION, YOLOv5, \
    ULTRALYTICS
from helpers.parameters import ModelParameters
from models.experimental import Ensemble
from models.yolo import Detect as YoloDetect
from models.yolo import Segment as YoloSegment
from torch import nn
from ultralytics.nn.modules.head import Detect as UltralyticsDetect
from ultralytics.nn.modules.head import Segment as UltralyticsSegment
from utils.downloads import attempt_download

import ultralytics


class PyTorchModelLoader:
    """ Class to load the Yolo pytorch model

    Attributes
    ----------
    model_input_path: str
        The path to the pytorch model

    input_resolution: int
        The input resolution for the model
    """

    def __init__(self, model_input_path, model_parameters=ModelParameters()):
        self.model_input_path = model_input_path
        self.model_parameters = model_parameters

    def load(self, fuse=True):
        """
        Loads the pytorch model

        Returns
        ----------
        self: PyTorchModelLoader
            A wrapper for the pytorch model, with all useful information
        """
        self.__load_model(fuse)
        self.__dry_run()
        self.__load_attributes()
        return self

    def __load_model(self, fuse):
        try:
            self.torch_model = self.attempt_load(self.model_input_path, device='cpu', inplace=True, fuse=fuse)
        except:
            self.torch_model = self.attempt_load(self.model_input_path)
        if isinstance(self.torch_model.names, list):
            self.torch_model.names = [name.encode().decode('ascii', 'ignore') for i, name in
                                      enumerate(self.torch_model.names)]
        else:
            self.torch_model.names = [name.encode().decode('ascii', 'ignore') for i, name in
                                      enumerate(self.torch_model.names.values())]

    def __dry_run(self):
        self.torch_model.eval()  # Will return predictions, model outputs
        sample_input = torch.zeros(
            (BATCH_SIZE, NB_CHANNEL, self.model_parameters.input_resolution, self.model_parameters.input_resolution))
        self.torch_model(sample_input)

    def __load_attributes(self):
        pretty_name = Path(self.torch_model.yaml.get('yaml_file', self.model_input_path)).stem.replace('yolo', 'YOLO')
        description = f'Ultralytics {pretty_name} model'

        self.model_parameters.nb_classes = len(self.torch_model.names)

        head = self.torch_model.model[-1]
        if isinstance(head, YoloSegment) or isinstance(head, UltralyticsSegment):
            self.model_parameters.model_type = SEGMENTATION
            self.number_of_masks = self.torch_model.model[-1].nm
        elif isinstance(head, YoloDetect) or isinstance(head, UltralyticsDetect):
            self.model_parameters.model_type = DETECTION
            self.number_of_masks = None
        else:
            raise ValueError(
                f"Yolo model should end with either 'Segment' or 'Detect' not {type(self.torch_model.model[-1])}")
        self.model_parameters.model_orig = YOLOv5 if isinstance(head, YoloDetect) or isinstance(head,
                                                                                                YoloSegment) else ULTRALYTICS

        self.model_parameters.metadata = {
            'description': description,
            'author': 'Ultralytics',
            'license': 'AGPL-3.0 https://ultralytics.com/license',
            'date': datetime.now().isoformat(),
            'version': ultralytics.__version__,
            'orig': self.model_parameters.model_orig,
            'stride': int(max(self.torch_model.stride)),
            'task': self.model_parameters.model_type,
            'imgsz': self.model_parameters.input_resolution,
            'names': ','.join(self.torch_model.names),
            'normalized': self.model_parameters.include_normalization,
            'max_det': self.model_parameters.max_det,
            'nms': self.model_parameters.include_nms
        }

    def attempt_load(self, weights, device=None, inplace=True, fuse=True):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        from models.yolo import Model

        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
            ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

            # Model compatibility updates
            if not hasattr(ckpt, 'stride'):
                ckpt.stride = torch.tensor([32.])
            if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
                ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
            ckpt.pt_path = w

            model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

        # Module compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (
                    nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, YoloDetect, YoloSegment, Model,
                    UltralyticsDetect,
                    UltralyticsSegment):
                m.inplace = inplace  # torch 1.7.0 compatibility
                if t in {YoloDetect, YoloSegment} and not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model
        if len(model) == 1:
            return model[-1]

        # Return detection ensemble
        print(f'Ensemble created with {weights}\n')
        for k in 'names', 'nc', 'yaml':
            setattr(model, k, getattr(model[0], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
        return model
