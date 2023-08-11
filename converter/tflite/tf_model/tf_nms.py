import tensorflow as tf
from helpers.constants import WH_SLICE, SCORE_SLICE, CLASSES_SLICE, DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, \
    YOLOv5, ULTRALYTICS, DEFAULT_MAX_NUMBER_DETECTION, DEFAULT_INPUT_RESOLUTION
from helpers.coordinates import tf_xywh2yxyx_yolo


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

    model_orig: str
        Whether the model is created with repo 'yolov5' or 'ultralytics'

    img_size: int
        The input size of an image
    """

    def __init__(self, max_det: int = DEFAULT_MAX_NUMBER_DETECTION,
                 iou_thres: float = DEFAULT_IOU_THRESHOLD,
                 conf_thres: float = DEFAULT_CONF_THRESHOLD,
                 model_orig: str = YOLOv5,
                 img_size: int = DEFAULT_INPUT_RESOLUTION):
        self.max_det = max_det
        self.iou_thres = iou_thres if type(iou_thres) is float else iou_thres[0]
        self.conf_thres = conf_thres if type(conf_thres) is float else conf_thres[0]
        self.model_orig = model_orig
        self.img_size = img_size

    def compute(self, predictions, nc: int):
        """ Compute NMS for detection

        Parameters
        ----------
        predictions
            The predictions made by the model

        nc: int
            The number of classes
        """
        for image_predictions in predictions:
            outputs = self.__nms_simple(image_predictions, nc)
            return outputs

    def compute_with_masks(self, model_output, nc: int, input_resolution: int):
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
        yxyx, classes, scores, nb_detected = self.__nms_simple(predictions[0], nc)

        if self.model_orig == ULTRALYTICS:
            predictions = tf.transpose(predictions[0], perm=(1, 0))
        else:
            predictions = predictions[0]

        if self.model_orig == ULTRALYTICS:
            masks = predictions[..., CLASSES_SLICE[0] - 1 + nc:]
        else:
            masks = predictions[..., CLASSES_SLICE[0] + nc:]
        masks = tf.gather(masks, self.nms, axis=0, batch_dims=1)
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
        # (nb_det, 1, mw)
        crop_mask1 = tf.math.multiply(tf.cast(tf.greater_equal(r, x1), dtype=y1.dtype),
                                      tf.cast(tf.less(r, x2), dtype=y1.dtype))
        # (nb_det, mh, 1)
        crop_mask2 = tf.math.multiply(tf.cast(tf.greater_equal(c, y1), dtype=y1.dtype),
                                      tf.cast(tf.less(c, y2), dtype=y1.dtype))
        # (nb_det, mh, mw)
        crop_mask = tf.math.multiply(crop_mask1, crop_mask2)
        masks = tf.math.multiply(masks, crop_mask)

        masks = tf.expand_dims(masks, axis=3)
        masks = tf.image.resize(masks, (input_resolution, input_resolution))
        masks = tf.clip_by_value(masks, clip_value_min=0.5, clip_value_max=1)
        masks = tf.squeeze(masks, axis=3)

        return yxyx, classes, scores, masks, nb_detected

    def __nms_simple(self, image_predictions, nc: int):
        if self.model_orig == ULTRALYTICS:
            image_predictions = tf.transpose(image_predictions, perm=(1, 0))

        # (1) Compute the scores (n, #classes)
        if self.model_orig == ULTRALYTICS:
            scores = image_predictions[:, CLASSES_SLICE[0] - 1:CLASSES_SLICE[0] - 1 + nc]
        else:
            scores = image_predictions[:, CLASSES_SLICE[0]:CLASSES_SLICE[0] + nc] * image_predictions[:,
                                                                                    SCORE_SLICE[0]:SCORE_SLICE[1]]

        # Convert xywh to y1, x1, y2, x2 (n, 4)
        yxyx = tf_xywh2yxyx_yolo(image_predictions[..., :WH_SLICE[1]])

        if self.model_orig == ULTRALYTICS:
            yxyx = yxyx / self.img_size  # done in the Detect for Yolov5

        # (2) Find the classes (unique label for now) (n,) and their respective scores (n,)
        classes = tf.argmax(scores, axis=1)
        scores = tf.gather(scores, classes, axis=1, batch_dims=1)

        # (3) Run NMS (#detected,)
        # Returns indices of detected boxes in ascending order, with 0 padding at the end, as well as the number of detected boxes
        nms, nb_detected = tf.image.non_max_suppression_padded(yxyx, scores, max_output_size=self.max_det,
                                                               iou_threshold=self.iou_thres,
                                                               score_threshold=self.conf_thres,
                                                               pad_to_max_output_size=True, sorted_input=False,
                                                               canonicalized_coordinates=True)
        self.nms = nms
        yxyx = tf.gather(yxyx, nms, axis=0, batch_dims=1)
        classes = tf.cast(tf.gather(classes, nms, axis=0, batch_dims=1), dtype=tf.float32)
        scores = tf.gather(scores, nms, axis=0, batch_dims=1)

        # Get the right output shape (1, d, 4), (1, d), (1, d), (1,)
        yxyx = tf.expand_dims(yxyx, axis=0)
        classes = tf.expand_dims(classes, axis=0)
        scores = tf.expand_dims(scores, axis=0)
        nb_detected = tf.cast(tf.expand_dims(nb_detected, axis=0), dtype=tf.float32)

        return yxyx, classes, scores, nb_detected
