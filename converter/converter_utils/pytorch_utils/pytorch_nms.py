import torchvision.ops
from helpers.constants import SCORE_SLICE, CLASSES_SLICE, WH_SLICE, DEFAULT_MAX_NUMBER_DETECTION, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, NORMALIZATION_FACTOR, ULTRALYTICS, YOLOv5
from helpers.coordinates import pt_xywh2yxyx_yolo
from torch import nn


class YoloNMS(nn.Module):
    """ Class to add NMS to the Yolo model

    Attributes
    ----------
    model: PyTorchModelLoader
        The pytorch model

    max_det: int
        The maximum number of detections

    iou_thres: float
        The IoU threshold

    conf_thres:
        The confidence threshold

    normalized: bool
        If the input should be normalized prior to be fed to the model

    nmsed: bool
        If NMS should be applied to the output of the model
    """

    def __init__(self, pt_model, max_det=DEFAULT_MAX_NUMBER_DETECTION, iou_thres=DEFAULT_IOU_THRESHOLD,
                 conf_thres=DEFAULT_CONF_THRESHOLD, normalized=True, nmsed=True, class_labels=None):
        super(YoloNMS, self).__init__()
        self.pt_model = pt_model
        self.names = class_labels if class_labels else pt_model.torch_model.names
        self.nc = len(self.names)
        self.max_det = max_det
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.normalized = normalized
        self.nmsed = nmsed

    def forward(self, x):
        """
        Runs the inference

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        yxyx, classes, scores, masks
            The detections made by the model
        """
        if self.normalized:
            # Normalize the input
            x /= NORMALIZATION_FACTOR
        # Pass through YOLOv5
        predictions = self.pt_model.torch_model(x)

        if isinstance(predictions, tuple):
            if len(predictions) == 3:
                images_predictions, protos, _ = predictions
            else:
                images_predictions, p = predictions
                if isinstance(p, tuple):
                    protos = p[-1]
                else:
                    protos = p
        else:
            images_predictions = predictions
            protos = None

        if not self.nmsed:
            return images_predictions, protos

        # Yolov5 output: torch.Size([1, 25200, 85]) / Yolov8: torch.Size([1, 84, 8400])
        return self.nms_yolov5(images_predictions, self.pt_model.model_parameters.model_orig), protos

    def nms_yolov5(self, images_predictions, model_orig):
        print(images_predictions.shape)
        # YOLOV5 (1, nb predictions, 4 + 1 + nb classes + nb masks)
        # ULTRALYTICS (1, 4 + nb classes + nb masks)
        # E.g. score does not exist anymore in new one.
        mi = 4 + self.nc  # mask start index

        if model_orig == ULTRALYTICS:
            candidates = images_predictions[:, 4:mi].amax(1) > self.conf_thres  # candidates
        else:
            candidates = images_predictions[..., SCORE_SLICE[0]] > self.conf_thres  # candidates
        if model_orig == ULTRALYTICS:
            images_predictions = images_predictions.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

        for i, image_prediction in enumerate(images_predictions):  # image index, image inference
            image_prediction = image_prediction[candidates[i]]  # confidence

            if model_orig == ULTRALYTICS:
                scores = image_prediction[:, CLASSES_SLICE[0] - 1:CLASSES_SLICE[0] - 1 + self.nc]
            else:
                scores = image_prediction[:, CLASSES_SLICE[0]:CLASSES_SLICE[0] + self.nc] * image_prediction[:,
                                                                                            SCORE_SLICE[0]:SCORE_SLICE[
                                                                                                1]]

            yxyx = pt_xywh2yxyx_yolo(image_prediction[:, :WH_SLICE[1]])  # xywh to yxyx

            scores, classes = scores.max(1, keepdim=True)
            scores = scores.reshape(scores.shape[0])
            classes = classes.reshape(classes.shape[0]).float()
            if model_orig == ULTRALYTICS:
                masks = image_prediction[:, CLASSES_SLICE[0] - 1 + self.nc:]
            else:
                masks = image_prediction[..., CLASSES_SLICE[0] + self.nc:]
            nms = torchvision.ops.nms(yxyx, scores, iou_threshold=self.iou_thres)
            return yxyx[nms].unsqueeze(0), classes[nms].unsqueeze(0), scores[nms].unsqueeze(0), masks[nms].unsqueeze(0)
