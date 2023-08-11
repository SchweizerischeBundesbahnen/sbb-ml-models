import ast
import json
import logging

import numpy as np
import tensorflow as tf
import torch
from helpers.constants import IMAGE_NAME, IOU_NAME, CONF_NAME, NORMALIZED_SUFFIX, QUANTIZED_SUFFIX, BOUNDINGBOX_NAME, \
    CLASSES_NAME, SCORES_NAME, NUMBER_NAME, PREDICTIONS_NAME, MASKS_NAME, PREDICTIONS_ULTRALYTICS_NAME, \
    DEFAULT_MAX_NUMBER_DETECTION, PROTOS_NAME
from helpers.constants import YOLOv5, ULTRALYTICS
from tf_model.tf_nms import NMS
from tflite_support import metadata as _metadata


class TFLiteModel:
    """ Class to load a TFLite model and run inference

    Attributes
    ----------
    model_path: str
        The path to the TFLite model
    """

    def __init__(self, model_path):
        logging.info("- Initializing TFLite model...")
        self.metadata_displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
        # Load labels
        self.labels = self.__load_labels()
        logging.info(f"- There are {len(self.labels)} labels.")
        # Load information from metadata
        self.output_order, self.input_order, self.normalized_image, self.do_nms, self.quantized, self.model_orig = self.__read_from_metadata()
        # Initialize the interepreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        # Get input size
        input_shape = self.input_details[self.input_order.index(IMAGE_NAME)]['shape']
        self.batch_size = int(input_shape[0])
        self.img_size = (int(input_shape[1]), int(input_shape[2]))
        logging.info(
            f"- The model takes {self.batch_size} image{'s' if self.batch_size > 1 else ''} of size {self.img_size} at a time.")
        logging.info(
            f"- It has {len(self.input_order)} input{'s' if len(self.input_order) > 1 else ''}: {', '.join(self.input_order)}")
        logging.info(
            f"- It has {len(self.output_order)} output{'s' if len(self.output_order) > 1 else ''}: {', '.join(self.output_order)}")

    def get_input_info(self):
        """
        Returns the information about the input

        Returns
        ----------
        normalized, img size, batch size, PIL image, channel first
            Information about the input
        """
        return self.normalized_image, self.img_size, self.batch_size, False, False  # Normalized, img size, batch size, PIL image, channel first

    def predict(self, img, iou_threshold, conf_threshold):
        """
        Runs the inference

        Parameters
        ----------
        img: torch.Tensor
            The input image

        iou_threshold: float
            The IoU threshold

        conf_threshold: float
            The confidence threshold

        Returns
        ----------
        yxyx, classes, scores, masks, nb_detected
            The detections made by the model
        """
        # Run inference for each image in the directory
        interpreter = self.interpreter
        input_details = self.input_details
        output_details = self.output_details
        output_order = self.output_order
        input_order = self.input_order
        do_nms = self.do_nms
        quantized = self.quantized

        if quantized:
            scale, zero_point = input_details[0]['quantization']
            img = img.numpy() / scale + zero_point
            img = img.astype(np.uint8)

        if do_nms:
            interpreter.set_tensor(input_details[input_order.index(IMAGE_NAME)]['index'], img)

            interpreter.invoke()

            if self.model_orig == YOLOv5:
                predictions_name = PREDICTIONS_NAME
            else:
                predictions_name = PREDICTIONS_ULTRALYTICS_NAME

            predictions = interpreter.get_tensor(output_details[output_order.index(predictions_name)]['index'])
            if quantized:
                scale, zero_point = output_details[output_order.index(predictions_name)]['quantization']
                predictions = predictions.astype(np.float32)
                predictions = (predictions - zero_point) * scale

            # Postprocessing
            nms = NMS(DEFAULT_MAX_NUMBER_DETECTION, [iou_threshold], [conf_threshold], self.model_orig)

            if PROTOS_NAME in output_order:
                protos = interpreter.get_tensor(output_details[output_order.index(PROTOS_NAME)]['index'])
                yxyx, classes, scores, masks, nb_detected = [x.numpy() for x in
                                                             nms.compute_with_masks((predictions, protos),
                                                                                    len(self.labels), self.img_size[0])]
                return yxyx, classes, scores, masks, nb_detected

            yxyx, classes, scores, nb_detected = [x.numpy() for x in nms.compute(predictions, len(self.labels))]
        else:

            interpreter.set_tensor(input_details[input_order.index(IMAGE_NAME)]['index'], img)
            if len(input_order) == 3:
                iou_input = [iou_threshold]
                conf_input = [conf_threshold]
                interpreter.set_tensor(input_details[input_order.index(IOU_NAME)]['index'],
                                       np.array(iou_input, dtype=np.float32))
                interpreter.set_tensor(input_details[input_order.index(CONF_NAME)]['index'],
                                       np.array(conf_input, dtype=np.float32))

            interpreter.invoke()

            yxyx = interpreter.get_tensor(output_details[output_order.index(BOUNDINGBOX_NAME)]['index'])
            classes = interpreter.get_tensor(output_details[output_order.index(CLASSES_NAME)]['index'])
            scores = interpreter.get_tensor(output_details[output_order.index(SCORES_NAME)]['index'])
            nb_detected = interpreter.get_tensor(output_details[output_order.index(NUMBER_NAME)]['index'])

            if MASKS_NAME in output_order:
                masks = interpreter.get_tensor(output_details[output_order.index(MASKS_NAME)]['index'])
                return yxyx, classes, scores, masks, nb_detected

        return yxyx, classes, scores, None, nb_detected

    def __load_labels(self):
        associated_files = self.metadata_displayer.get_packed_associated_file_list()
        if 'temp_meta.txt' in associated_files:
            labels = \
            ast.literal_eval(self.metadata_displayer.get_associated_file_buffer('temp_meta.txt').decode('utf-8'))[
                'names']
        else:
            raise ValueError("The model does not contain the metadata: temp_meta.txt.")
        return labels

    def __read_from_metadata(self):
        json_file = self.metadata_displayer.get_metadata_json()
        metadata = json.loads(json_file)
        # Read input order
        input_metadata = metadata['subgraph_metadata'][0]['input_tensor_metadata']

        input_order = [input['name'] if IMAGE_NAME not in input['name'] else IMAGE_NAME for input in input_metadata]

        normalized_image = False
        quantized_image = False
        # Check the inputs
        expected_input_name = [IOU_NAME, CONF_NAME, IMAGE_NAME]
        for i, input_name in enumerate(input_order):
            if input_name not in expected_input_name:
                raise ValueError(f'Unknown input: {input_name}')
            expected_input_name.remove(input_name)
            if input_name == IMAGE_NAME:
                image_name = metadata['subgraph_metadata'][0]['input_tensor_metadata'][i]['name']  # Get original name
                if NORMALIZED_SUFFIX in image_name:
                    normalized_image = True
                    logging.info("- The model expects normalized images [0, 1].")
                if QUANTIZED_SUFFIX in image_name:
                    quantized_image = True
                    logging.info("- The model uses full integer quantization.")

        # Read output order
        output_metadata = metadata['subgraph_metadata'][0]['output_tensor_metadata']
        output_order = [output['name'] for output in output_metadata]
        model_orig = None
        # Check the outputs
        if len(output_order) == 4:
            # Detection
            expected_output_name = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, NUMBER_NAME]
            for output_name in output_order:
                if output_name not in expected_output_name:
                    raise ValueError(f'Unknown output: {output_name}')
                expected_output_name.remove(output_name)
            do_nms = False
        elif len(output_order) == 5:
            # Segmentation
            expected_output_name = [BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, NUMBER_NAME, MASKS_NAME]
            for output_name in output_order:
                if output_name not in expected_output_name:
                    raise ValueError(f'Unknown output: {output_name}')
                expected_output_name.remove(output_name)
            do_nms = False
        elif len(output_order) == 2:
            # Segmentation
            expected_output_name = [PREDICTIONS_NAME, PREDICTIONS_ULTRALYTICS_NAME, PROTOS_NAME]
            for output_name in output_order:
                if output_name not in expected_output_name:
                    raise ValueError(f'Unknown output: {output_name}')
                expected_output_name.remove(output_name)
            do_nms = True
            model_orig = ULTRALYTICS if PREDICTIONS_ULTRALYTICS_NAME in output_order else YOLOv5
        else:
            if output_order[0] not in [PREDICTIONS_NAME, PREDICTIONS_ULTRALYTICS_NAME]:
                raise ValueError(f'Unknown output: {output_order[0]}')
            do_nms = True
            model_orig = ULTRALYTICS if output_order[0] == PREDICTIONS_ULTRALYTICS_NAME else YOLOv5

        return output_order, input_order, normalized_image, do_nms, quantized_image, model_orig
