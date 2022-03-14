from copy import deepcopy
from pathlib import Path
import logging
from models.tf import parse_model

from constants import NB_CHANNEL, NORMALIZATION_FACTOR
from tf_model.tf_nms import NMS
from tf_utils.parameters import ModelParameters, PostprocessingParameters


class tf_Model():
    def __init__(self, ch=NB_CHANNEL, nc=None, pt_model=None, model_parameters=ModelParameters(),
                 postprocessing_parameters=PostprocessingParameters(),
                 tf_raw_resize=False):  # model, input channels, number of classes
        super(tf_Model, self).__init__()

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
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=pt_model,
                                                imgsz=model_parameters.img_size)  # model, savelist, ch_out
        self.model_parameters = model_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def predict_nms(self, image, iou_thres, conf_thres):
        predictions = self.predict(image)
        # Add TensorFlow NMS
        return NMS(self.postprocessing_parameters, iou_thres, conf_thres).compute(predictions)

    def predict(self, image):
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

        return x[0]  # output only first tensor [1,25200,51] = [xywh, conf, class0, class1, ...]
