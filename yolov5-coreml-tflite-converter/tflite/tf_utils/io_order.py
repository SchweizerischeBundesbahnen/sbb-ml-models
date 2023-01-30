from typing import List

import tensorflow as tf

from helpers.constants import IMAGE_NAME, IOU_NAME, CONF_NAME, BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, NUMBER_NAME, \
    PREDICTIONS_NAME, MASKS_NAME, SEGMENTATION


class IOOrder:
    """ Class to get the output order

    Attributes
    ----------
    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. include normalization, nms)
    """

    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    def get_output_order(self, interpreter: tf.lite.TFLiteConverter) -> List[str]:
        """ Returns the output order from the TFLite interpreter

        Parameters
        ----------
        interpreter: tf.lite.TFLiteConverter
            The TFLite converter

        Returns
        ----------
        output_order: List[str]
            The output names in order
        """
        if self.model_parameters.include_nms:
            # FIXME: ugly fix because the output order is not always the same but the names are consistent
            # To get the names: convert a model and open the converted model with Netron
            # Look at the outputs and map them to the following:
            #   BOUNDINGBOX_NAME should have 3 dimensions e.g. (1, 20, 4)
            #   CLASSES_NAME should have 2 dimensions e.g. (1, 20)
            #   SCORES_NAME should have 2 dimensions e.g. (1, 20)
            #   NUMBER_NAME should have 1 dimension e.g. (1)
            # To differentiate between CLASSES_NAME and SCORE_NAME, take a look at the end of the model
            #   The SCORES_NAME should come from NonMaxSuppressionV4 -> Reshape -> Gather -> Reshape
            #   The CLASSES_NAME should come from NonMaxSuppressionV4 -> Reshape -> Gather -> Cast -> Reshape
            if self.model_parameters.model_type == SEGMENTATION:
                output_map = {'StatefulPartitionedCall:0': NUMBER_NAME, 'StatefulPartitionedCall:1': MASKS_NAME,
                              'StatefulPartitionedCall:2': BOUNDINGBOX_NAME, 'StatefulPartitionedCall:3': CLASSES_NAME,
                              'StatefulPartitionedCall:4': SCORES_NAME}
            else:
                output_map = {'StatefulPartitionedCall:0': NUMBER_NAME, 'StatefulPartitionedCall:1': BOUNDINGBOX_NAME,
                              'StatefulPartitionedCall:2': CLASSES_NAME, 'StatefulPartitionedCall:3': SCORES_NAME}

            output_details = interpreter.get_output_details()
            try:
                output_order = [output_map[output['name']] for output in output_details]
            except KeyError as e:
                raise KeyError(f"{e}, the outputs are: {output_details}")
        else:
            output_order = [PREDICTIONS_NAME]
        return output_order

    def get_input_order(self, interpreter) -> List[str]:
        """ Returns the input order from the TFLite interpreter

        Parameters
        ----------
        interpreter: tf.lite.TFLiteConverter
            The TFLite converter

        Returns
        ----------
        input_order: List[str]
            The input names in order
        """
        if self.model_parameters.include_nms:
            # FIXME: ugly fix because the input order is not always the same but the names are consistent
            # To get the names: convert and model and open the converted model with Netron
            # Look at the inputs and map them to the following:
            #   IMAGE_NAME should have 4 dimensions e.g. (1, 640, 640, 3)
            #   IOU_NAME should have 1 dimension e.g. (1)
            #   CONF_NAME should have 1 dimension e.g. (1)
            # To differentiate between IOU_NAME and CONF_NAME, take a look at the end of the model
            #   In NonMaxSuppressionV4, look at name of input 4
            #       Find where it comes from, and this input should be the IOU_NAME
            #   Similarly, look at input 5 of NonMaxSuppressionV4
            #       Find where it comes from, and this is the CONF_NAME
            # Another way is trial and error: arbitrarily chose one to be IOU_NAME and the other to be CONF_NAME
            #   Convert a new model with these updated names
            #   The inference with conf = 1, no detections should be made
            #   The inference with iou = 0, no overlap between the detections
            #   The inference with conf = 0, iou = 1, the maximum number of detections should be made, i.e. 20
            # If that is not the case, exchange IOU_NAME and CONF_NAME
            input_map = {'serving_default_input_1:0': IMAGE_NAME, 'serving_default_input_2:0': IOU_NAME,
                         'serving_default_input_3:0': CONF_NAME, 'input_1': IMAGE_NAME, 'input_2': IOU_NAME,
                         'input_3': CONF_NAME}

            input_details = interpreter.get_input_details()
            try:
                input_order = [input_map[input['name']] for input in input_details]
            except KeyError as e:
                raise KeyError(f"{e}, the inputs are: {input_details}")
        else:
            input_order = [IMAGE_NAME]
        return input_order
