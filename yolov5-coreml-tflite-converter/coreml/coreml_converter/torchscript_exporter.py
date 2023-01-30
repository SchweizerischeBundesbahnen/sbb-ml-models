import logging

import torch
from torch.jit import ScriptModule

from helpers.constants import BLUE, END_COLOR, GREEN, RED


class TorchscriptExporter:
    """ Class to export a model to TorchScript

    Attributes
    ----------
    model: ModelWrapper
        The model to export to TorchScript
    """

    def __init__(self, model):
        self.model = model

    def export(self) -> ScriptModule:
        """
        Traces a pytorch model and produces a TorchScript

        Returns
        ----------
        ts: ScriptModule
            The trace (torchscript) for the given PyTorch model
        """

        sample_input = torch.zeros(self.model.input_shape)
        check_inputs = [(torch.rand(*self.model.input_shape),),
                        (torch.rand(*self.model.input_shape),)]

        try:
            logging.info(f'{BLUE}Starting TorchScript export with torch {torch.__version__}...{END_COLOR}')
            ts = torch.jit.trace(self.model.torch_model, sample_input, check_inputs=check_inputs)
            logging.info(f'{GREEN}TorchScript export success{END_COLOR}')
            return ts
        except Exception as e:
            raise Exception(f'{RED}TorchScript export failure:{END_COLOR} {e}')
