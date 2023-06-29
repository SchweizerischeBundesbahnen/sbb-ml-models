import logging
from copy import deepcopy
from pathlib import Path

from tf_model.tf_nms import NMS
from tf_utils.parameters import ModelParameters, PostprocessingParameters

from helpers.constants import NB_CHANNEL, NORMALIZATION_FACTOR, DEFAULT_CONF_THRESHOLD, SEGMENTATION, \
    DEFAULT_IOU_THRESHOLD
from models.tf import parse_model


class TFModel:
    """
    Class to convert Pytorch modules into TF modules

    Parameters
    ----------
    ch: int
        The number of channels

    nc: int
        The number of classes

    pt_model: models.yolo.DetectionModel or models.yolo.SegmentationModel
        The Pytorch model to convert

    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. include normalization, nms)

    postprocessing_parameters: PostprocessingParameters
        The parameters for the postprocesssing (if any) (e.g. nms type, ...)

    """

    def __init__(self, ch: int = NB_CHANNEL, nc: int = None, pt_model=None,
                 model_parameters: ModelParameters = ModelParameters(),
                 postprocessing_parameters: PostprocessingParameters = PostprocessingParameters()):
        super(TFModel, self).__init__()

        self.pt_model = pt_model
        cfg = pt_model.yaml
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            logging.info('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value

        # Parse the Pytorch model, and create an equivalent TF model
        self.model, self.savelist, self.nc = parse_model(deepcopy(self.yaml), ch=[ch], model=pt_model,
                                                         imgsz=(model_parameters.input_resolution,
                                                                model_parameters.input_resolution))
        self.model_parameters = model_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def predict(self, image, iou_thres: float = DEFAULT_IOU_THRESHOLD, conf_thres: float = DEFAULT_CONF_THRESHOLD):
        """ Runs the inference on an image

        Parameters
        ----------
        image
            The input to the network

        iou_thres: float
            The IoU threshold, if nms is included in the model

        conf_thres: float
            The confidence threshold, if nms is included in the model

        Returns
        ----------
        location, category, score, number of detections
            if the model is a DETECTION model and includes NMS
        location, category, score, masks, number of detections
            if the model is a SEGMENTATION model and includes NMS
        predictions
            if the model is a DETECTION model without NMS
        predictions, protos
            if the model is a SEGMENTATION model without NMS
        """
        y = []  # outputs
        x = image

        # Normalize the input
        if self.model_parameters.include_normalization:
            x /= NORMALIZATION_FACTOR

        for i, m in enumerate(self.model.layers):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # predictions: [1, nb predictions, number of classes + 5]
        # protos: [1, masks height, masks width, number of masks]
        # DETECTION: returns predictions
        # SEGMENTATION: returns predictions, protos

        if self.model_parameters.include_nms:
            if self.model_parameters.model_type == SEGMENTATION:
                # returns location, category, score, masks
                return NMS(self.postprocessing_parameters, iou_thres, conf_thres).compute_with_masks(x, self.nc,
                                                                                                     self.model_parameters.input_resolution)
            else:
                # returns location, category, score
                return NMS(self.postprocessing_parameters, iou_thres, conf_thres).compute(x, self.nc)

        return x
