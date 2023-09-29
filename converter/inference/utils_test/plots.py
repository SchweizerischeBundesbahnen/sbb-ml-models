import cv2

from helpers.coordinates import scale_coords_yolo, pt_yxyx2xyxy_yolo
from utils.plots import Annotator, Colors
import random

WHITE_COLOR = (225, 255, 255)

def get_red():
    redval = random.randint(180, 255)
    greenval = random.randint(0, 100)
    blueval = random.randint(0, 100)
    return (blueval, greenval, redval)

def get_orange():
    redval = random.randint(180, 255)
    greenval = random.randint(50, redval - 60)
    blueval = random.randint(50, greenval)
    return (redval, greenval, blueval)

def get_green():
    greenval = random.randint(100, 255)
    redval = random.randint(20, greenval - 60)
    blueval = random.randint(redval - 20, redval + 20)
    return (redval, greenval, blueval)

DATASET_COLORS_31 = [get_red() if i in [0] else get_orange() if i in [2, 15, 16] else get_green() for i in range(31)]
DATASET_COLORS_10 = [get_red() if i in [0] else get_green() for i in range(10)]


def plot_boxes(img_size, img_origs, yxyxs, classes, scores, nb_detecteds, labels):
    colors = Colors()
    nb_detected = int(nb_detecteds[0])
    yxyx = yxyxs[0][:nb_detected]
    classe = classes[0][:nb_detected]
    score = scores[0][:nb_detected]
    img_orig = img_origs[0]

    orig_h, orig_w = img_orig.shape[:2]
    yxyx = scale_coords_yolo(img_orig, img_size, yxyx)

    line_thickness = int(round(0.0005 * (orig_h + orig_w) / 2) + 1)
    annotator = Annotator(img_orig, line_width=line_thickness, example=str(classes))

    # Plot bounding boxes
    for j in range(nb_detected - 1, -1, -1):
        xyxy = pt_yxyx2xyxy_yolo(yxyx[j])
        label = labels[int(classe[j])]
        label_score = label + f' {score[j]:0.3}'
        annotator.box_label(xyxy, label_score, colors(int(classe[j])))
    return img_orig


def plot_masks(img_size, img_origs, yxyxs, classes, scores, masks, nb_detecteds, labels):
    colors = Colors()
    nb_detected = int(nb_detecteds[0])
    yxyx = yxyxs[0][:nb_detected]
    classe = classes[0][:nb_detected]
    score = scores[0][:nb_detected]
    img_orig = img_origs[0]
    masks = masks[:nb_detected]

    orig_h, orig_w = img_orig.shape[:2]
    yxyx = scale_coords_yolo(img_orig, img_size, yxyx)

    line_thickness = int(round(0.0005 * (orig_h + orig_w) / 2) + 1)
    annotator = Annotator(img_orig, line_width=line_thickness, example=str(classes))

    if len(labels) == 31:
        masks_colors = [DATASET_COLORS_31[int(i)] for i in classe]
    elif len(labels) == 10:
        masks_colors = [DATASET_COLORS_10[int(i)] for i in classe]
    else:
        masks_colors = [colors(int(classe[i])) for i in range(len(classe))]

    # Plot the masks
    annotator.masks(masks, colors=masks_colors)

    colors = Colors()
    # Plot bounding boxes
    for j in range(nb_detected - 1, -1, -1):
        xyxy = pt_yxyx2xyxy_yolo(yxyx[j])
        label = labels[int(classe[j])]
        label_score = label + f' {score[j]:0.3}'
        annotator.box_label(xyxy, label_score, colors(int(classe[j])))
    return img_orig


def plot_boxes_xywh(img_origs, xywhs, classes, scores, nb_detecteds, labels):
    # xy top left (i.e. coco format) - coordinates of original images
    colors = Colors()
    nb_detected = int(nb_detecteds[0])
    xywh = xywhs[0][:nb_detected]
    classe = classes[0][:nb_detected]
    score = scores[0][:nb_detected]
    img_orig = img_origs[0]
    orig_h, orig_w = img_orig.shape[:2]

    # Plot bounding boxes
    for j in range(nb_detected - 1, -1, -1):
        label = labels[int(classe[j])]

        x1 = xywh[j][0] * orig_w
        x2 = (xywh[j][2] + xywh[j][0]) * orig_w
        y1 = xywh[j][1] * orig_h
        y2 = (xywh[j][3] + xywh[j][1]) * orig_h

        # Add bounding box
        line_thickness = int(round(0.0005 * (orig_h + orig_w) / 2) + 1)
        c1 = (int(x1), int(y1))  # top left coordinates
        c2 = (int(x2), int(y2))  # bottom right coordinates

        cv2.rectangle(img_orig, c1, c2, color=colors(int(classe[j]), True), thickness=line_thickness,
                      lineType=cv2.LINE_AA)

        # Add label
        font_thickness = max(line_thickness - 1, 1)
        label_score = label + f' {score[j]:0.3}'
        text_size = cv2.getTextSize(label_score, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + text_size[0], c1[1] - text_size[1] - 3
        cv2.rectangle(img_orig, c1, c2, color=colors(int(classe[j]), True), thickness=-1,
                      lineType=cv2.LINE_AA)  # filled
        cv2.putText(img_orig, label_score, (c1[0], c1[1] - 2), 0, line_thickness / 3, WHITE_COLOR,
                    thickness=line_thickness, lineType=cv2.LINE_AA)
