import logging
import time

import tensorflow as tf
from tensorflow import keras
from tf_model.tf_model import TFModel
from tf_utils.parameters import ModelParameters, PostprocessingParameters

from helpers.constants import BATCH_SIZE, NB_CHANNEL, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, GREEN, BLUE, \
    END_COLOR, RED
from models.tf import TFDetect


class KerasModel:
    """ Class to create a Keras model, given a PyTorch one

    Attributes
    ----------
    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. type, use nms, ...)

    postprocessing_parameters: PostprocessingParameters
        The parameters for the postprocesssing (if any) (e.g. nms type, ...)
    """

    def __init__(self, model_parameters=ModelParameters(), postprocessing_parameters=PostprocessingParameters()):
        self.model_parameters = model_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def create(self, pt_model, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
               conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> keras.Model:
        """ Create Keras model from a Pytorch model

        Parameters
        ----------
        pt_model: models.yolo.DetectionModel or models.yolo.SegmentationModel
            The Pytorch model to convert into Keras

        iou_threshold: float
            The IoU threshold, if hard-coded

        conf_threshold: float
            The confidence threshold, if hard-coded

        """
        logging.info(f"{BLUE}Creating the keras model...{END_COLOR}")
        start_time = time.time()
        # Get Keras model
        tf_model = TFModel(nc=self.model_parameters.nb_classes, pt_model=pt_model,
                           model_parameters=self.model_parameters,
                           postprocessing_parameters=self.postprocessing_parameters)

        m = tf_model.model.layers[-1]
        assert isinstance(m, TFDetect), f"{RED}Keras model creation failure:{END_COLOR} the last layer must be Detect"
        m.training = False

        # NHWC Input for TensorFlow
        img = tf.zeros(
            (BATCH_SIZE, self.model_parameters.input_resolution, self.model_parameters.input_resolution,
             NB_CHANNEL))  # image size(1, 640, 640, 3)

        y = tf_model.predict(img)  # dry run

        image_input = keras.Input(
            shape=(self.model_parameters.input_resolution, self.model_parameters.input_resolution, NB_CHANNEL),
            batch_size=BATCH_SIZE)
        if self.model_parameters.include_nms:
            if self.model_parameters.include_threshold:
                # Input: image
                # DETECTION output: location, category, score, number of detections
                # SEGMENTATION output: location, category, score, masks, number of detections
                inputs = image_input
                outputs = tf_model.predict(inputs, [iou_threshold], [conf_threshold])
            else:
                # Input: image, iou threshold, conf threshold
                # DETECTION output: location, category, score, number of detections
                # SEGMENTATION output: location, category, score, masks, number of detections
                iou_input = keras.Input(batch_shape=(BATCH_SIZE,))
                conf_input = keras.Input(batch_shape=(BATCH_SIZE,))
                inputs = (image_input, iou_input, conf_input)

                outputs = tf_model.predict(image_input, iou_input, conf_input)
        else:
            # Input: image
            # DETECTION output: predictions
            # SEGMENTATION output: predictions, protos
            inputs = image_input
            predictions = tf_model.predict(inputs)
            outputs = predictions

        keras_model = keras.Model(inputs=inputs, outputs=outputs)
        keras_model.summary()

        end_time = time.time()
        logging.info(
            f"{GREEN}Keras model creation success:{END_COLOR} it took {int(end_time - start_time)} seconds to create the Keras model.")
        return keras_model
