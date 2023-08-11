import tensorflow as tf
import torch


def tf_xywh2yxyx_yolo(xywh):
    # Convert nx4 boxes from [x, y, w, h] (x,y center) to [y1, x1, y2, x2] (xy1=top-left, xy2=bottom-right)
    x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
    return tf.concat([y - h / 2, x - w / 2, y + h / 2, x + w / 2], axis=-1)


def pt_xywh2yxyx_yolo(xywh):
    # Convert nx4 boxes from [x, y, w, h] (x,y center) to [y1, x1, y2, x2] (xy1=top-left, xy2=bottom-right)
    xywh = xywh if isinstance(xywh, torch.Tensor) else torch.from_numpy(xywh)
    yxyx = xywh.clone()
    yxyx[:, 0] = xywh[:, 1] - xywh[:, 3] / 2  # y1 = y - h/2
    yxyx[:, 1] = xywh[:, 0] - xywh[:, 2] / 2  # x1 = x - w/2
    yxyx[:, 2] = xywh[:, 1] + xywh[:, 3] / 2  # y2 = y + h/2
    yxyx[:, 3] = xywh[:, 0] + xywh[:, 2] / 2  # x2 = x + w/2
    return yxyx


def pt_yxyx2xyxy_yolo(yxyx):
    # Convert nx4 boxes [y1, x1, y2, x2] (xy1=top-left, xy2=bottom-right) to [x1, y1, x2, y2] (xy1=top-left, xy2=bottom-right)
    yxyx = yxyx if isinstance(yxyx, torch.Tensor) else torch.from_numpy(yxyx)
    xyxy = yxyx.clone()
    xyxy[..., 0] = yxyx[..., 1]
    xyxy[..., 1] = yxyx[..., 0]
    xyxy[..., 2] = yxyx[..., 3]
    xyxy[..., 3] = yxyx[..., 2]
    return xyxy


def pt_yxyx2xywh_coco(yxyx):
    # Convert nx4 boxes [y1, x1, y2, x2] (xy1=top-left, xy2=bottom-right) to [x, y, w, h] (xy top-left)
    yxyx = yxyx if isinstance(yxyx, torch.Tensor) else torch.from_numpy(yxyx)
    y = yxyx.clone()
    y[..., 0] = yxyx[..., 1]  # x = x1
    y[..., 1] = yxyx[..., 0]  # y = y1
    y[..., 2] = torch.abs(yxyx[..., 3] - yxyx[..., 1])  # w = x2 - x1
    y[..., 3] = torch.abs(yxyx[..., 2] - yxyx[..., 0])  # h = y2 - y1
    return y


def pt_normalize_xywh(xywh, img):
    h, w = img.shape[:2]
    xywh = xywh if isinstance(xywh, torch.Tensor) else torch.from_numpy(xywh)
    y = xywh.clone()
    y[..., 0] = xywh[..., 0] / w
    y[..., 1] = xywh[..., 1] / h
    y[..., 2] = xywh[..., 2] / w
    y[..., 3] = xywh[..., 3] / h
    return y


def scale_coords_yolo(img_orig, img_size, yxyx):
    orig_h, orig_w = img_orig.shape[:2]
    new_h, new_w = img_size

    # Compute ratio (orig_shape * ratio = new_shape)
    gain_ratio = min(new_h / orig_h, new_w / orig_w)
    pad_h, pad_w = (new_h - orig_h * gain_ratio) / 2, (new_w - orig_w * gain_ratio) / 2

    yxyx = yxyx if isinstance(yxyx, torch.Tensor) else torch.from_numpy(yxyx)
    y = yxyx.clone()

    # Rescale coordinates x1, y1, x2, x2 from the padded, resized image to the original image
    y[..., 1] = torch.clip((yxyx[..., 1] * new_w - pad_w) / gain_ratio, 0, orig_w)
    y[..., 3] = torch.clip((yxyx[..., 3] * new_w - pad_w) / gain_ratio, 0, orig_w)

    y[..., 0] = torch.clip((yxyx[..., 0] * new_h - pad_h) / gain_ratio, 0, orig_h)
    y[..., 2] = torch.clip((yxyx[..., 2] * new_h - pad_h) / gain_ratio, 0, orig_h)
    return y
