import kwcoco
from pathlib import Path


def fix_coordinate_system(value: float, min_value: float = 0.0, max_value: float = 1.0):
    """
    In case of Out-Of-Range values: fix the rounding error
    """

    value = min(value, max_value)
    value = max(value, min_value)
    return value


def convertRelCoco2Yolo(x: float, y: float, w: float, h: float):
    """
    INPUT: relative coco parameters for bounding box
    OUTPUT: relative yolov5 parameters for bounding box
    parameters:
    origin: top left, 0,0
    x: horizontal relative position to the top left corner
    y: vertical relative position to the top left corner
    w: relative width of the bounding box
    h: relative height of the bounding box

    return: relative position
    """
    x = fix_coordinate_system(x)
    y = fix_coordinate_system(y)
    w = fix_coordinate_system(w)
    h = fix_coordinate_system(h)

    x_yolo = x + w / 2  # division has higher precedence than addition
    y_yolo = y + h / 2

    w_yolo = w
    h_yolo = h

    return (x_yolo, y_yolo, w_yolo, h_yolo)


def convertAbsCoco2relCoco(x, y, w, h, img_w, img_h):
    """
    INPUT: coco absolute parameters, image width, image height (in pixels)
    OUTPUT: coco relative parameters
    """
    x_rel = x / img_w
    y_rel = y / img_h
    w_rel = w / img_w
    h_rel = h / img_h
    return (x_rel, y_rel, w_rel, h_rel)


def convertAbsCoco2Yolo(x, y, w, h, img_w, img_h):
    """
    """
    x_rel, y_rel, w_rel, h_rel = convertAbsCoco2relCoco(
        x, y, w, h, img_w, img_h)
    return convertRelCoco2Yolo(x_rel, y_rel, w_rel, h_rel)
