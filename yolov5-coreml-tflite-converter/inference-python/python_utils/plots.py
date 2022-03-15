import cv2

from coordinates import scale_coords_yolo

WHITE_COLOR = (225, 255, 255)


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def plot_boxes(img_size, img_origs, yxyxs, classes, scores, nb_detecteds, labels):
    colors = Colors()
    nb_detected = int(nb_detecteds[0])
    yxyx = yxyxs[0][:nb_detected]
    classe = classes[0][:nb_detected]
    score = scores[0][:nb_detected]
    img_orig = img_origs[0]

    orig_h, orig_w = img_orig.shape[:2]
    yxyx = scale_coords_yolo(img_orig, img_size, yxyx)

    # Plot bounding boxes
    for j in range(nb_detected - 1, -1, -1):
        label = labels[int(classe[j])]

        # Add bounding box
        line_thickness = int(round(0.0005 * (orig_h + orig_w) / 2) + 1)
        c1 = (int(yxyx[j][1]), int(yxyx[j][0]))  # top left coordinates
        c2 = (int(yxyx[j][3]), int(yxyx[j][2]))  # bottom right coordinates

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


def plot_boxes_xywh(img_origs, xywhs, classes, scores, nb_detecteds, labels):
    # xy top left (i.e. coco format) - coordinates of original images
    colors = Colors()
    nb_detected = int(nb_detecteds[0])
    xywh = xywhs[0][:nb_detected]
    classe = classes[0][:nb_detected]
    score = scores[0][:nb_detected]
    img_orig = img_origs[0]
    orig_h, orig_w = img_orig.shape[:2]

    print(nb_detected)

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
