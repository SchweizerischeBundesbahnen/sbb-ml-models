import tensorflow as tf

from helpers.constants import WH_SLICE, SCORE_SLICE, CLASSES_SLICE, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    SIMPLE, \
    PADDED, COMBINED
from helpers.coordinates import tf_xywh2yxyx_yolo
from tf_utils.parameters import PostprocessingParameters


class NMS:
    """ Class to compute NMS

    Attributes
    ----------
    postprocessing_parameters: PostprocessingParameters
        The parameters for the postprocesssing (if any) (e.g. nms type, ...)

    iou_thres: float
        The IoU threshold

    conf_thres: float
        The confidence threshold
    """

    def __init__(self, postprocessing_parameters: PostprocessingParameters = PostprocessingParameters(),
                 iou_thres: float = DEFAULT_IOU_THRESHOLD,
                 conf_thres: float = DEFAULT_CONF_THRESHOLD):
        self.max_det = postprocessing_parameters.max_det
        self.iou_thres = iou_thres if type(iou_thres) is float else iou_thres[0]
        self.conf_thres = conf_thres if type(conf_thres) is float else conf_thres[0]
        self.nms_type = postprocessing_parameters.nms_type

    def compute(self, predictions, nc: int):
        """ Compute NMS for detection

        Parameters
        ----------
        predictions
            The predictions made by the model

        nc: int
            The number of classes
        """
        if self.nms_type == COMBINED:
            outputs = self.__nms_combined(predictions, nc)
            return outputs
        else:
            for image_predictions in predictions:
                outputs = self.__nms_simple(image_predictions, nc)
                return outputs

    def compute_with_masks(self, model_output, nc: int):
        """ Compute NMS for segmentation

        Parameters
        ----------
        model_output
            The output of the YOLO model (predictions, protos)

        nc: int
            The number of classes
        """
        predictions = model_output[0]
        protos = model_output[1]
        # NMS combined does not work for segmentation at the moment
        yxyx, classes, scores, nb_detected = self.__nms_simple(predictions[0], nc)

        masks = predictions[0][..., CLASSES_SLICE[0] + nc:]
        masks = tf.gather(masks, self.nms, axis=0, batch_dims=1)
        if self.nms_type == SIMPLE:
            masks = masks[:nb_detected[0]]

        protos = tf.transpose(protos[0], perm=(2, 0, 1))

        # Process masks
        c, mh, mw = protos.shape  # CHW
        n = masks.shape[0]  # Number of detections
        # (nb_det, c) @ (c, mh*mw) = (nb_det, mh*mw))
        masks = tf.linalg.matmul(masks, tf.reshape(protos, (c, -1)))
        # reshape (nb_det, mh, mw)
        masks = tf.reshape(masks, (-1, mh, mw))

        # scale yxyx to the mask dim
        y1, x1, y2, x2 = tf.split(tf.reshape(yxyx, (n, 1, -1)), num_or_size_splits=4, axis=-1)
        y1 = tf.scalar_mul(tf.cast(mh, dtype=y1.dtype), y1)
        x1 = tf.scalar_mul(tf.cast(mw, dtype=y1.dtype), x1)
        y2 = tf.scalar_mul(tf.cast(mh, dtype=y1.dtype), y2)
        x2 = tf.scalar_mul(tf.cast(mw, dtype=y1.dtype), x2)

        # crop mask
        r = tf.reshape(tf.range(mw, dtype=y1.dtype), (1, 1, mw))
        c = tf.reshape(tf.range(mw, dtype=y1.dtype), (1, mh, 1))
        crop_mask1 = tf.math.multiply(tf.cast(tf.greater_equal(r, x1), dtype=y1.dtype),
                                      tf.cast(tf.less(r, x2), dtype=y1.dtype))
        crop_mask2 = tf.math.multiply(tf.cast(tf.greater_equal(c, y1), dtype=y1.dtype),
                                      tf.cast(tf.less(c, y2), dtype=y1.dtype))
        crop_mask = tf.math.multiply(crop_mask1, crop_mask2)
        masks = tf.math.multiply(masks, crop_mask)

        return yxyx, classes, scores, masks, nb_detected

    def __nms_combined(self, image_predictions, nc: int):
        # (1) Compute the scores (n, #classes)
        scores = image_predictions[..., CLASSES_SLICE[0]:CLASSES_SLICE[0] + nc] * image_predictions[...,
                                                                                  SCORE_SLICE[0]:SCORE_SLICE[1]]

        # Convert xywh to y1, x1, y2, x2 (n, 4)
        yxyx = tf.expand_dims(tf_xywh2yxyx_yolo(image_predictions[..., :WH_SLICE[1]]), axis=2)

        # (2) Run NMS (#detected,)
        # Returns indices of detected boxes in ascending order, with 0 padding at the end, as well as the number of detected boxes
        nms = tf.image.combined_non_max_suppression(yxyx, scores, max_total_size=self.max_det,
                                                    max_output_size_per_class=self.max_det,
                                                    iou_threshold=self.iou_thres, score_threshold=self.conf_thres)

        return nms.nmsed_boxes, nms.nmsed_classes, nms.nmsed_scores, nms.valid_detections

    def __nms_simple(self, image_predictions, nc: int):
        if self.nms_type == SIMPLE:
            candidates = image_predictions[..., SCORE_SLICE[0]] > self.conf_thres
            image_predictions = image_predictions[candidates]

        # (1) Compute the scores (n, #classes)
        scores = image_predictions[:, CLASSES_SLICE[0]:CLASSES_SLICE[0] + nc] * image_predictions[:,
                                                                                SCORE_SLICE[0]:SCORE_SLICE[1]]
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

        self.nms = nms
        yxyx = tf.gather(yxyx, nms, axis=0, batch_dims=1)
        classes = tf.cast(tf.gather(classes, nms, axis=0, batch_dims=1), dtype=tf.float32)
        scores = tf.gather(scores, nms, axis=0, batch_dims=1)

        if self.nms_type == SIMPLE:
            yxyx = yxyx[:nb_detected]
            classes = classes[:nb_detected]
            scores = scores[:nb_detected]

        # Get the right output shape (1, d, 4), (1, d), (1, d), (1, d, 32), (1,)
        yxyx = tf.expand_dims(yxyx, axis=0)
        classes = tf.expand_dims(classes, axis=0)
        scores = tf.expand_dims(scores, axis=0)
        nb_detected = tf.cast(tf.expand_dims(nb_detected, axis=0), dtype=tf.float32)

        return yxyx, classes, scores, nb_detected
