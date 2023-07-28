# disabling PyLint docstring, no self using
# pylint: disable=C0115,C0116,R0201
from src.utils.coordinates_util import convertRelCoco2Yolo, fix_coordinate_system


class TestConvert:
    def test_convert_rel_coco_2_yolo_accepts_float(self):
        """ check  """
        result = convertRelCoco2Yolo(0.0, 0.0, 0.0, 0.0)
        assert result == (0, 0, 0, 0)

    def test_convert_rel_coco_2_yolo_accepts_integer(self):
        assert convertRelCoco2Yolo(int(0), int(
            0), int(0), int(0)) == (0, 0, 0, 0)

    def convert_rel_coco_2_yolo_example(self):
        assert convertRelCoco2Yolo(0.1, 0.2, 0.3, 0.4) == (0.25, 0.4, 0.3, 0.4)

    def convert_rel_coco_2_yolo_example2(self):
        assert convertRelCoco2Yolo(0.2, 0.3, 0.4, 0.5) == (0.4, 0.5, 0.4, 0.5)

    def test_rounding_error_valid_values(self):
        for x in [0, 0.01, 0.001, 0.3, 0.999, 1]:
            assert fix_coordinate_system(x) == x

    def test_rounding_error_invalid_values(self):
        for x in [[-1, 0], [-0.01, 0], [-0.001, 0], [1.01, 1.0], [1.2, 1.0], [2, 1.0]]:
            assert fix_coordinate_system(x[0]) == x[1]

    def test_rounding_error_custom_values(self):
        for x in [[-0.5, -0.5, -1, 1], [-0.01, 0, 0, 5], [-6, -5, -5, 10], [10.5, 10, 5, 10], [-4, -5, -10, -5],
                  [11, 10, 1, 10]]:
            assert fix_coordinate_system(x[0], x[2], x[3]) == x[1]
