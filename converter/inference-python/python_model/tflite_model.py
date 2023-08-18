import ast
import json
import logging

import numpy as np
import tensorflow as tf
from helpers.constants import IMAGE_NAME, IOU_NAME, CONF_NAME, BOUNDINGBOX_NAME, \
    CLASSES_NAME, SCORES_NAME, NUMBER_NAME, PREDICTIONS_NAME, MASKS_NAME, \
    DEFAULT_MAX_NUMBER_DETECTION, PROTOS_NAME, SEGMENTATION, BLUE, END_COLOR
from python_model.inference_model_abs import InferenceModelAbs
from tf_model.tf_nms import NMS
from tflite_support import metadata as _metadata


class TFLiteModel(InferenceModelAbs):
    """ Class to load a TFLite model and run inference

        Attributes
        ----------
        model_path: str
            The path to the TFLite model
    """
    def __init__(self, model_path):
        super().__init__()
        logging.info(f"{BLUE}Initializing TFLite model...{END_COLOR}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.model_path = model_path
        self.interpreter.allocate_tensors()
        # Load metadata
        self.__load_metadata()

    def predict(self, img, iou_threshold, conf_threshold):
        # Run inference for each image in the directory
        if self.do_nms:
            self.interpreter.set_tensor(self.input_details[self.input_order.index(IMAGE_NAME)]['index'], img)
            self.interpreter.invoke()

            predictions = self.interpreter.get_tensor(
                self.output_details[self.output_order.index(PREDICTIONS_NAME)]['index'])

            # Postprocessing
            nms = NMS(DEFAULT_MAX_NUMBER_DETECTION, [iou_threshold], [conf_threshold], self.model_orig)

            if self.model_type == SEGMENTATION:
                protos = self.interpreter.get_tensor(self.output_details[self.output_order.index(PROTOS_NAME)]['index'])

                yxyx, classes, scores, masks, nb_detected = [x.numpy() for x in
                                                             nms.compute_with_masks((predictions, protos),
                                                                                    len(self.labels), self.img_size[0])]
                return yxyx, classes, scores, masks, nb_detected

            yxyx, classes, scores, nb_detected = [x.numpy() for x in nms.compute(predictions, len(self.labels))]
        else:
            self.interpreter.set_tensor(self.input_details[self.input_order.index(IMAGE_NAME)]['index'], img)
            if not self.do_nms:
                iou_input = [iou_threshold]
                conf_input = [conf_threshold]
                self.interpreter.set_tensor(self.input_details[self.input_order.index(IOU_NAME)]['index'],
                                            np.array(iou_input, dtype=np.float32))
                self.interpreter.set_tensor(self.input_details[self.input_order.index(CONF_NAME)]['index'],
                                            np.array(conf_input, dtype=np.float32))

            self.interpreter.invoke()

            yxyx = self.interpreter.get_tensor(self.output_details[self.output_order.index(BOUNDINGBOX_NAME)]['index'])
            classes = self.interpreter.get_tensor(self.output_details[self.output_order.index(CLASSES_NAME)]['index'])
            scores = self.interpreter.get_tensor(self.output_details[self.output_order.index(SCORES_NAME)]['index'])
            nb_detected = self.interpreter.get_tensor(
                self.output_details[self.output_order.index(NUMBER_NAME)]['index'])

            if MASKS_NAME in self.output_order:
                masks = self.interpreter.get_tensor(self.output_details[self.output_order.index(MASKS_NAME)]['index'])
                return yxyx, classes, scores, masks, nb_detected

        return yxyx, classes, scores, None, nb_detected

    def __load_metadata(self):
        # Read from metadata file
        metadata_displayer = _metadata.MetadataDisplayer.with_model_file(self.model_path)
        associated_files = metadata_displayer.get_packed_associated_file_list()
        if 'temp_meta.txt' in associated_files:
            metadata = \
                ast.literal_eval(metadata_displayer.get_associated_file_buffer('temp_meta.txt').decode('utf-8'))
        else:
            raise ValueError("The model does not contain the metadata: temp_meta.txt.")

        self.model_type = metadata['task']
        self.model_orig = metadata['orig']
        self.labels = metadata['names']
        self.do_normalize = not metadata['normalized']
        self.do_nms = not metadata['nms']
        self.quantization_type = metadata['quantization_type']

        self.pil_image = False
        self.channel_first = False

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Read from metadata model
        json_file = metadata_displayer.get_metadata_json()
        metadata = json.loads(json_file)

        # Read input order
        input_metadata = metadata['subgraph_metadata'][0]['input_tensor_metadata']
        self.input_order = [input['name'] if IMAGE_NAME not in input['name'] else IMAGE_NAME for input in input_metadata]

        # Read output order
        output_metadata = metadata['subgraph_metadata'][0]['output_tensor_metadata']
        self.output_order = [output['name'] for output in output_metadata]

        # Get input size
        input_shape = self.input_details[self.input_order.index(IMAGE_NAME)]['shape']
        self.batch_size = int(input_shape[0])
        self.img_size = (int(input_shape[1]), int(input_shape[2]))

        self.input_dict = { name : self.input_details[i]['shape'] for i, name in enumerate(self.input_order) }
        self.output_dict = { name : self.output_details[i]['shape'] for i, name in enumerate(self.output_order) }