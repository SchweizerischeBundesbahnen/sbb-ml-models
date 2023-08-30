import logging
import shutil
import time
from copy import deepcopy
from pathlib import Path

import tensorflow as tf
from helpers.constants import NB_CHANNEL, DEFAULT_CONF_THRESHOLD, SEGMENTATION, GREEN, \
    DEFAULT_IOU_THRESHOLD, YOLOv5, NORMALIZATION_FACTOR, BLUE, RED, END_COLOR
from models.tf import parse_model
from tf_model.tf_nms import NMS
from helpers.parameters import ModelParameters
from ultralytics.engine.model import Model


class SavedKerasModel(tf.keras.layers.Layer):
    def __init__(self, fn):
        super(SavedKerasModel, self).__init__()
        self.fn = fn

    def call(self, inputs):
        return self.fn(inputs)


class TFModel:
    """
    Class to convert Pytorch modules into TF modules

    Parameters
    ----------
    pt_model: models.yolo.DetectionModel or models.yolo.SegmentationModel
        The Pytorch model to convert

    model_input_path: str
        The path to the pytorch model

    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. include normalization, nms)
    """

    def __init__(self, pt_model=None,
                 model_input_path: str = None,
                 model_parameters: ModelParameters = ModelParameters()):
        super(TFModel, self).__init__()
        self.model_parameters = model_parameters

        if model_parameters.model_orig == YOLOv5:
            try:
                logging.info(f'{BLUE}Building Keras model from Pytorch one...{END_COLOR}')
                cfg = pt_model.yaml
                if isinstance(cfg, dict):
                    self.yaml = cfg  # model dict
                else:  # is *.yaml
                    import yaml  # for torch hub
                    self.yaml_file = Path(cfg).name
                    with open(cfg) as f:
                        self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

                # Define model
                if self.model_parameters.nb_classes != self.yaml['nc']:
                    logging.info('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
                    self.yaml['nc'] = nc  # override yaml value

                # Parse the Pytorch model, and create an equivalent TF model
                self.model, self.savelist, self.model_parameters.nb_classes = parse_model(deepcopy(self.yaml),
                                                                                          ch=[NB_CHANNEL],
                                                                                          model=pt_model,
                                                                                          imgsz=(
                                                                                              model_parameters.input_resolution,
                                                                                              model_parameters.input_resolution))
            except TypeError as e:
                raise TypeError(
                    f'{RED}Tensorflow conversion failure: The pytorch model may contain unsupported layers.{END_COLOR} {e}.')
            except Exception as e:
                raise Exception(f'{RED}Tensorflow conversion failure:{END_COLOR} {e}')
        else:
            try:
                logging.info(f'{BLUE}Exporting Pytorch model to TensorFlow SavedModel...{END_COLOR}')
                start_time = time.time()
                # First convert to saved_model
                model = Model(model_input_path)
                model.model.model[-1].format = 'tflite'
                model.model.model[-1].export = True
                saved_model_path = model.export(format="saved_model")
                model = tf.saved_model.load(saved_model_path)
                end_time = time.time()
                logging.info(
                    f"{GREEN}TensorFlow SavedModel export success{END_COLOR}: it took {int(end_time - start_time)} seconds to create export to SavedModel.")
            except Exception as e:
                raise Exception(f'{RED}TensorFlow SavedModel export failure:{END_COLOR} {e}')

            onnx_model_path = str(saved_model_path).replace("_saved_model", ".onnx")
            logging.info(f'{BLUE}Wrapping TensorFlow SavedModel in Keras model...{END_COLOR}')
            self.model = SavedKerasModel(model)
            logging.info(f'{BLUE}Removing {saved_model_path}, {onnx_model_path}{END_COLOR}')
            shutil.rmtree(saved_model_path)
            Path(onnx_model_path).unlink()

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

        # Run the inference
        if self.model_parameters.model_orig == YOLOv5:
            for i, m in enumerate(self.model.layers):
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.savelist else None)  # save output

        else:

            x = self.model(x)

        # output x
        # predictions YOLOv5: [1, nb predictions, number of classes + 5 (bbox + score)]
        # predictions Ultralytics: [1, number of classes + 4 (bbox) + number of masks, nb predictions]
        # protos: [1, masks height, masks width, number of masks]
        # DETECTION: returns predictions
        # SEGMENTATION: returns predictions, protos

        if self.model_parameters.include_nms:
            if self.model_parameters.model_type == SEGMENTATION:
                # returns location, category, score, masks
                return NMS(self.model_parameters.max_det, iou_thres, conf_thres,
                           self.model_parameters.model_orig).compute_with_masks(x, self.model_parameters.nb_classes,
                                                                                self.model_parameters.input_resolution)
            else:
                # returns location, category, score
                return NMS(self.model_parameters.max_det, iou_thres, conf_thres,
                           self.model_parameters.model_orig).compute(x, self.model_parameters.nb_classes)
        return x
