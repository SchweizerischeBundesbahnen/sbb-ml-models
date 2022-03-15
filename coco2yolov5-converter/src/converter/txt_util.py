# pylint: disable=C0114,C0115,C0116,R0201
from pathlib import Path


class TxtUtil:
    """ Text Writer Util """

    @staticmethod
    def write_labels_txt_file(label_rows: list, txt_file: Path):
        """ Write annotations to text file
            'label_rows' must be a list of type [int,float,float,float,float]
            where:
                0: <int>    id of class
                1: <float>  x_center
                2: <float>  y_center
                3: <float>  width
                4: <float>  height

            From official YOLOv5 documentation: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
            One row per object
            Each row is class x_center y_center width height format.
            Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
            Class numbers are zero-indexed (start from 0).     
        """
        precision = 5
        # More than 5 is in sub-pixel range: Typical img resolution is: 768px
        lines = [
            f"{label[0]} {round(label[1], precision)} {round(label[2], precision)} {round(label[3], precision)} {round(label[4], precision)}\n" for label in label_rows]
        with txt_file.open('w') as file:
            file.writelines(lines)
