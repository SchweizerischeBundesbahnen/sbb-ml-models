import torchvision.ops
from torch import nn

from constants import SCORE_SLICE, CLASSES_SLICE, WH_SLICE, DEFAULT_MAX_NUMBER_DETECTION, DEFAULT_IOU_THRESHOLD, \
    DEFAULT_CONF_THRESHOLD, NORMALIZATION_FACTOR
from coordinates import pt_xywh2yxyx_yolo


class YoloNMS(nn.Module):
    def __init__(self, model, max_det=DEFAULT_MAX_NUMBER_DETECTION, iou_thres=DEFAULT_IOU_THRESHOLD,
                 conf_thres=DEFAULT_CONF_THRESHOLD, normalized=True, nmsed=True):
        super(YoloNMS, self).__init__()
        self.model = model
        self.names = model.names
        self.max_det = max_det
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.normalized = normalized
        self.nmsed = nmsed

    def forward(self, x):
        if self.normalized:
            # Normalize the input
            x /= NORMALIZATION_FACTOR

        # Pass through YOLOv5
        images_predictions = self.model(x)[0]

        if not self.nmsed:
            return images_predictions

        candidates = images_predictions[..., SCORE_SLICE[0]] > self.conf_thres

        for i, image_predictions in enumerate(images_predictions):
            image_predictions = image_predictions[candidates[i]]

            # Compute the scores (n, #classes)
            scores = image_predictions[:, CLASSES_SLICE[0]:] * image_predictions[:, SCORE_SLICE[0]:SCORE_SLICE[1]]

            # Convert xywh to y1, x1, y2, x2 (n, 4)
            yxyx = pt_xywh2yxyx_yolo(image_predictions[..., :WH_SLICE[1]])

            scores, classes = scores.max(1, keepdim=True)
            scores = scores.reshape(scores.shape[0])
            classes = classes.reshape(classes.shape[0]).float()

            # nms = torch
            nms = torchvision.ops.nms(yxyx, scores, iou_threshold=self.iou_thres)
            # if nms.shape[0] > self.max_det:
            # nms = nms[:self.max_det]

            return yxyx[nms].unsqueeze(0), classes[nms].unsqueeze(0), scores[nms].unsqueeze(0)
