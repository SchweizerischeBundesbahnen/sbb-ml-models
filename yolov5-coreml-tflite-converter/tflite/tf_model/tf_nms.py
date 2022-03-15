import tensorflow as tf

from constants import WH_SLICE, SCORE_SLICE, CLASSES_SLICE, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, SIMPLE, \
    PADDED, COMBINED
from coordinates import tf_xywh2yxyx_yolo
from tf_utils.parameters import PostprocessingParameters


class NMS:
    def __init__(self, postprocessing_parameters=PostprocessingParameters(), iou_thres=DEFAULT_IOU_THRESHOLD,
                 conf_thres=DEFAULT_CONF_THRESHOLD):
        self.max_det = postprocessing_parameters.max_det
        self.iou_thres = iou_thres if type(iou_thres) is float else iou_thres[0]
        self.conf_thres = conf_thres if type(conf_thres) is float else conf_thres[0]
        self.nms_type = postprocessing_parameters.nms_type

    def compute(self, predictions):
        if self.nms_type == COMBINED:
            yxyx, classes, scores, nb_detected_out = self.nms_combined(predictions)
            return yxyx, classes, scores, nb_detected_out
        else:
            for image_predictions in predictions:
                yxyx, classes, scores, nb_detected_out = self.nms_simple(image_predictions)
                return yxyx, classes, scores, nb_detected_out

    def nms_combined(self, image_predictions):
        # (1) Compute the scores (n, #classes)
        scores = image_predictions[..., CLASSES_SLICE[0]:] * image_predictions[..., SCORE_SLICE[0]:SCORE_SLICE[1]]

        # Convert xywh to y1, x1, y2, x2 (n, 4)
        yxyx = tf.expand_dims(tf_xywh2yxyx_yolo(image_predictions[..., :WH_SLICE[1]]), axis=2)

        # (2) Run NMS (#detected,)
        # Returns indices of detected boxes in ascending order, with 0 padding at the end, as well as the number of detected boxes
        nms = tf.image.combined_non_max_suppression(yxyx, scores, max_total_size=self.max_det,
                                                    max_output_size_per_class=self.max_det,
                                                    iou_threshold=self.iou_thres, score_threshold=self.conf_thres)

        return nms.nmsed_boxes, nms.nmsed_classes, nms.nmsed_scores, nms.valid_detections

    def nms_simple(self, image_predictions):
        if self.nms_type == SIMPLE:
            candidates = image_predictions[..., SCORE_SLICE[0]] > self.conf_thres
            image_predictions = image_predictions[candidates]

        # (1) Compute the scores (n, #classes)
        scores = image_predictions[:, CLASSES_SLICE[0]:] * image_predictions[:, SCORE_SLICE[0]:SCORE_SLICE[1]]

        # Convert xywh to y1, x1, y2, x2 (n, 4)
        yxyx = tf_xywh2yxyx_yolo(image_predictions[..., :WH_SLICE[1]])

        # (2) Find the classes (unique label for now) (n,) and their respective scores (n,)
        classes = tf.argmax(scores, axis=1)
        scores = tf.gather(scores, classes, axis=1, batch_dims=1)

        # (3) Run NMS (#detected,)
        if self.nms_type == PADDED:
            # Returns indices of detected boxes in ascending order, with 0 padding at the end, as well as the number of detected boxes
            nms, nb_detected = tf.image.non_max_suppression_padded(yxyx, scores, max_output_size=self.max_det,
                                                                   iou_threshold=self.iou_thres,
                                                                   score_threshold=self.conf_thres,
                                                                   pad_to_max_output_size=True, sorted_input=False,
                                                                   canonicalized_coordinates=True)
        else:
            # Returns indices of detected boxes in ascending order, with 0 padding at the end
            nms = tf.image.non_max_suppression(yxyx, scores, max_output_size=self.max_det,
                                               iou_threshold=self.iou_thres, score_threshold=self.conf_thres)

            # Now we must retrieve the number of detected boxes
            nb_detected = tf.shape(tf.unique(nms).y)[0]

        yxyx = tf.gather(yxyx, nms, axis=0, batch_dims=1)
        classes = tf.cast(tf.gather(classes, nms, axis=0, batch_dims=1), dtype=tf.float32)
        scores = tf.gather(scores, nms, axis=0, batch_dims=1)

        if self.nms_type == SIMPLE:
            yxyx = yxyx[:nb_detected]
            classes = classes[:nb_detected]
            scores = scores[:nb_detected]

        # Get the right output shape (1, d, 4), (1, d), (1, d), (1,)
        yxyx = tf.expand_dims(yxyx, axis=0)
        classes = tf.expand_dims(classes, axis=0)
        scores = tf.expand_dims(scores, axis=0)
        nb_detected = tf.cast(tf.expand_dims(nb_detected, axis=0), dtype=tf.float32)

        return yxyx, classes, scores, nb_detected
