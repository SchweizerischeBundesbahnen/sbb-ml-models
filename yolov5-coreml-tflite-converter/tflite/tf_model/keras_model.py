import time

import tensorflow as tf
from models.tf import TFDetect
from tensorflow import keras

from constants import BATCH_SIZE, NB_CHANNEL, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, GREEN, BLUE, END_COLOR, RED
from tf_model.tf_model import tf_Model
from tf_utils.parameters import ModelParameters, PostprocessingParameters
import logging

class KerasModel:
    def __init__(self, model_parameters=ModelParameters(), postprocessing_parameters=PostprocessingParameters()):
        self.model_parameters = model_parameters
        self.postprocessing_parameters = postprocessing_parameters

    def create(self, pt_model, iou_threshold=DEFAULT_IOU_THRESHOLD,
               conf_threshold=DEFAULT_CONF_THRESHOLD):
        '''
        Create Keras model, based on Pytorch model
        '''
        logging.info(f"{BLUE}Creating the keras model...{END_COLOR}")
        start_time = time.time()
        # Get Keras model
        tf_model = tf_Model(nc=self.model_parameters.nb_classes, pt_model=pt_model,
                            model_parameters=self.model_parameters,
                            postprocessing_parameters=self.postprocessing_parameters,
                            tf_raw_resize=self.model_parameters.tf_raw_resize)

        m = tf_model.model.layers[-1]
        assert isinstance(m, TFDetect), f"{RED}Keras model creation failure:{END_COLOR} the last layer must be Detect"
        m.training = False

        # NHWC Input for TensorFlow
        img = tf.zeros(
            (BATCH_SIZE, *self.model_parameters.img_size, NB_CHANNEL))  # image size(1, 640, 640, 3)
        if self.model_parameters.include_nms:
            iou = tf.zeros(BATCH_SIZE)
            conf = tf.zeros(BATCH_SIZE)
            y = tf_model.predict_nms(img, iou, conf)  # dry run
        else:
            y = tf_model.predict(img)  # dry run

        image_input = keras.Input(shape=(*self.model_parameters.img_size, NB_CHANNEL), batch_size=BATCH_SIZE)
        if self.model_parameters.include_nms:
            if self.model_parameters.include_threshold:
                # Input: image
                # Output: location, category, score, number of detections
                inputs = image_input
                outputs = tf_model.predict_nms(inputs, [iou_threshold], [conf_threshold])
            else:
                # Input: image, iou threshold, conf threshold
                # Output: location, category, score, number of detections
                iou_input = keras.Input(batch_shape=(BATCH_SIZE,))
                conf_input = keras.Input(batch_shape=(BATCH_SIZE,))
                inputs = (image_input, iou_input, conf_input)
                outputs = tf_model.predict_nms(*inputs)
        else:
            # Input: image
            # Output: prediction
            inputs = image_input
            predictions = tf_model.predict(inputs)
            outputs = predictions

        keras_model = keras.Model(inputs=inputs, outputs=outputs)
        keras_model.summary()

        end_time = time.time()
        logging.info(
            f"{GREEN}Keras model creation success:{END_COLOR} it took {int(end_time - start_time)} seconds to create the Keras model.")
        return keras_model
