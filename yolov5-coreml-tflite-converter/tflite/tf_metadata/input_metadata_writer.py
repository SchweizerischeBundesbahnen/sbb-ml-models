from tflite_support import metadata_schema_py_generated as _metadata_fb

from helpers.constants import IMAGE_NAME, IOU_NAME, CONF_NAME, NORMALIZED_SUFFIX, QUANTIZED_SUFFIX
from tf_metadata.metadata_utils import MetadataHelper


class InputMetadataWriter(MetadataHelper):
    def __init__(self, input_order, img_size, normalized, quantized, multiple_inputs=False):
        self.input_order = input_order
        self.img_size = img_size
        self.normalized = normalized
        self.quantized = quantized
        self.multiple_inputs = multiple_inputs

    def write(self):
        image_meta = self.__create_image_meta()
        if self.multiple_inputs:
            if len(self.input_order) != 3:
                raise ValueError(
                    f"Expected 3 inputs ({IMAGE_NAME}, {IOU_NAME}, {CONF_NAME}) but got {len(self.input_order)} input{'s' if len(self.input_order) > 1 else ''} ({', '.join(self.input_order)})")
            iou_meta = self.__create_iou_meta()
            conf_meta = self.__create_conf_meta()

            input_map = {IMAGE_NAME: image_meta, IOU_NAME: iou_meta, CONF_NAME: conf_meta}
            input_metadata = [input_map[output_name] for output_name in self.input_order]
            return input_metadata
        else:
            if len(self.input_order) != 1:
                ValueError(
                    f"Expected 1 input ({IMAGE_NAME}) but got {len(self.input_order)} input{'s' if len(self.input_order) > 1 else ''} ({', '.join(self.input_order)})")
            return [image_meta]

    def __create_image_meta(self):
        image_meta = _metadata_fb.TensorMetadataT()
        # Input name
        image_meta.name = IMAGE_NAME
        self._add_content_image(image_meta)

        if self.quantized:
            # Fully quantized
            image_meta.name += QUANTIZED_SUFFIX
        if self.normalized:
            image_meta.description = (
                f"Input image to be classified. The expected image is {self.img_size[0]} x {self.img_size[1]}, with "
                "three channels (red, blue, and green) per pixel. Each value must be between 0 and 255.")
            self._add_normalization(image_meta, 0, 1)
            self._add_stats(image_meta, 255, 0)
        else:
            # Input should be normalized
            image_meta.name += NORMALIZED_SUFFIX
            image_meta.description = (
                f"Input image to be classified. The expected image is {self.img_size[0]} x {self.img_size[1]}, with "
                "three channels (red, blue, and green) per pixel. The values should be normalized and between 0 and 1.")
            self._add_normalization(image_meta, 0, 255)
            self._add_stats(image_meta, 1.0, 0.0)
        return image_meta

    def __create_iou_meta(self):
        iou_meta = _metadata_fb.TensorMetadataT()
        iou_meta.name = IOU_NAME
        iou_meta.description = "The IoU threshold for NMS"
        self._add_content_feature(iou_meta)
        self._add_stats(iou_meta, 1.0, 0.0)
        return iou_meta

    def __create_conf_meta(self):
        conf_meta = _metadata_fb.TensorMetadataT()
        conf_meta.name = CONF_NAME
        conf_meta.description = "The confidence threshold for NMS"
        self._add_content_feature(conf_meta)
        self._add_stats(conf_meta, 1.0, 0.0)
        return conf_meta
