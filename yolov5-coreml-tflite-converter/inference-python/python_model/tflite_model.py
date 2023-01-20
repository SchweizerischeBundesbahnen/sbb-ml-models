import json
import logging

import numpy as np
import tensorflow as tf
from tflite_support import metadata as _metadata

from helpers.constants import IMAGE_NAME, IOU_NAME, CONF_NAME, NORMALIZED_SUFFIX, QUANTIZED_SUFFIX, BOUNDINGBOX_NAME, \
    CLASSES_NAME, SCORES_NAME, NUMBER_NAME, PREDICTIONS_NAME, SIMPLE, MASKS_NAME
from tf_model.tf_nms import NMS
from tf_utils.parameters import PostprocessingParameters
import torch.nn.functional as F

class TFLiteModel:
    def __init__(self, model_path):
        logging.info("- Initializing TFLite model...")
        self.metadata_displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
        # Load labels
        self.labels = self.__load_labels()
        logging.info(f"- There are {len(self.labels)} labels.")
        # Load information from metadata
        self.output_order, self.input_order, self.normalized_image, self.do_nms, self.quantized = self.__read_from_metadata()
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
        return self.normalized_image, self.img_size, self.batch_size, False, False  # Normalized, img size, batch size, PIL image, channel first

    def get_labels(self):
        return self.labels

    def predict(self, img, iou_threshold, conf_threshold):
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

            predictions = interpreter.get_tensor(output_details[0]['index'])

            if quantized:
                scale, zero_point = output_details[0]['quantization']
                predictions = predictions.astype(np.float32)
                predictions = (predictions - zero_point) * scale

            # Postprocessing
            nms = NMS(PostprocessingParameters(nms_type=SIMPLE), [iou_threshold], [conf_threshold])
            yxyx, classes, scores, nb_detected = nms.compute(predictions)
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
                import torch
                masks = torch.Tensor(interpreter.get_tensor(output_details[output_order.index(MASKS_NAME)]['index']))
                masks = F.interpolate(masks[None], self.img_size, mode='bilinear', align_corners=False)[0]  # CHW
                return yxyx, classes, scores, masks.gt_(0.5).numpy(), nb_detected

        return yxyx, classes, scores, None, nb_detected

    def __load_labels(self):
        associated_files = self.metadata_displayer.get_packed_associated_file_list()

        if 'labels.txt' in associated_files:
            labels = self.metadata_displayer.get_associated_file_buffer('labels.txt').decode().split('\n')[:-1]
        else:
            raise ValueError("The model does not contain the file labels.txt.")
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

        output_metadata = metadata['subgraph_metadata'][0]['output_tensor_metadata']
        output_order = [output['name'] for output in output_metadata]
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
        else:
            if output_order[0] != PREDICTIONS_NAME:
                raise ValueError(f'Unknown output: {output_order[0]}')
            do_nms = True

        return output_order, input_order, normalized_image, do_nms, quantized_image
