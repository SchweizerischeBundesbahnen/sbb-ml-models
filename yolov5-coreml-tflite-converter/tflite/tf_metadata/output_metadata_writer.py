import os

from tflite_support import metadata_schema_py_generated as _metadata_fb
from typing import List

from helpers.constants import BOUNDINGBOX_NAME, CLASSES_NAME, SCORES_NAME, NUMBER_NAME, DETECTIONS_NAME, PREDICTIONS_NAME, MASKS_NAME
from tf_metadata.metadata_utils import MetadataHelper


class OutputMetadataWriter(MetadataHelper):
    """ Class that writes the output metadata for a TFLite model

    Attributes
    ----------
    output_order: List[str]
        The names of the outputs in order

    labels_path: str
        The path to the labels file

    nb_labels: int
        The number of classes

    max_det: int
        The maximum number of detections

    multiple_outputs: bool
        Whether the model has several outputs
    """
    def __init__(self, output_order: List[str], labels_path:str, nb_labels:int, max_det:int, multiple_outputs:bool=False):
        self.output_order = output_order
        self.labels_path = labels_path
        self.nb_labels = nb_labels
        self.max_det = max_det
        self.multiple_outputs = multiple_outputs

    def write(self):
        """ Write the output metadata """
        if self.multiple_outputs:
            # 4 for detections, 5 for segmentations
            if len(self.output_order) not in [4, 5]:
                raise ValueError(
                    f"Expected 4 output ({BOUNDINGBOX_NAME}, {CLASSES_NAME}, {SCORES_NAME}, {NUMBER_NAME}) but got {len(self.output_order)} output{'s' if len(self.output_order) > 1 else ''} ({', '.join(self.output_order)})")
            yxyx_meta = self.__create_yxyx_meta()
            class_meta = self.__create_class_meta()
            score_meta = self.__create_score_meta()
            nb_detected_meta = self.__create_nb_detected_meta()

            group = _metadata_fb.TensorGroupT()
            group.name = DETECTIONS_NAME
            if len(self.output_order) == 5:
                # For the segmentation
                masks_meta = self.__create_masks_meta()
                group.tensorNames = [yxyx_meta.name, class_meta.name, score_meta.name, masks_meta.name]
                output_map = {BOUNDINGBOX_NAME: yxyx_meta, CLASSES_NAME: class_meta,
                              SCORES_NAME: score_meta, NUMBER_NAME: nb_detected_meta, MASKS_NAME: masks_meta}
            else:
                # For the detection
                group.tensorNames = [yxyx_meta.name, class_meta.name, score_meta.name]
                output_map = {BOUNDINGBOX_NAME: yxyx_meta, CLASSES_NAME: class_meta,
                              SCORES_NAME: score_meta, NUMBER_NAME: nb_detected_meta}

            output_metadata = [output_map[output_name] for output_name in self.output_order]
            output_group = [group]
            return output_metadata, output_group
        else:
            # Predictions
            if len(self.output_order) != 1:
                raise ValueError(
                    f"Expected 1 output ({PREDICTIONS_NAME}) but got {len(self.output_order)} output{'s' if len(self.output_order) > 1 else ''} ({', '.join(self.output_order)})")
            predictions_meta = self.__create_prediction_meta()
            return [predictions_meta], None

    def __create_prediction_meta(self):
        # Creates the metadata for the predictions
        predictions_meta = _metadata_fb.TensorMetadataT()
        predictions_meta.name = PREDICTIONS_NAME
        predictions_meta.description = "The predictions made by each grid cell of the model on which one needs to run NMS."
        self._add_content_feature(predictions_meta)
        self._add_stats(predictions_meta, 1.0, 0)
        self.__add_labels_file(predictions_meta)
        return predictions_meta

    def __create_yxyx_meta(self):
        # Creates the metadata for the bounding boxes
        yxyx_meta = _metadata_fb.TensorMetadataT()
        yxyx_meta.name = BOUNDINGBOX_NAME
        yxyx_meta.description = "The bounding boxes coordinates (x1, y1) upper left, (x2, y2) bottom right (normalized to input image - resized)."
        self._add_content_bounding_box(yxyx_meta)
        self._add_range(yxyx_meta)
        self._add_stats(yxyx_meta, 1.0, 0.0)
        return yxyx_meta

    def __create_class_meta(self):
        # Creates the metadata for the classes
        class_meta = _metadata_fb.TensorMetadataT()
        class_meta.name = CLASSES_NAME
        class_meta.description = "The class corresponding to each bounding box."
        self._add_content_feature(class_meta)
        self._add_range(class_meta)
        self._add_stats(class_meta, self.nb_labels, 0)
        self.__add_labels_file(class_meta)
        return class_meta

    def __create_score_meta(self):
        # Creates the metadata for the scores
        score_meta = _metadata_fb.TensorMetadataT()
        score_meta.name = SCORES_NAME
        score_meta.description = "The confidence score corresponding to each bounding box."
        self._add_content_feature(score_meta)
        self._add_range(score_meta)
        self._add_stats(score_meta, 1.0, 0.0)
        return score_meta

    def __create_nb_detected_meta(self):
        # Creates the metadata for the number of detections
        nb_detected_meta = _metadata_fb.TensorMetadataT()
        nb_detected_meta.name = NUMBER_NAME
        nb_detected_meta.description = "The number of detected bounding boxes."
        self._add_content_feature(nb_detected_meta)
        self._add_stats(nb_detected_meta, self.max_det, 0.0)
        return nb_detected_meta

    def __create_masks_meta(self):
        # Creates the metadata for the masks
        masks_meta = _metadata_fb.TensorMetadataT()
        masks_meta.name = MASKS_NAME
        masks_meta.description = "The masks used for segmentation."
        self._add_content_feature(masks_meta)
        self._add_stats(masks_meta, 1.0, 0.0)
        return masks_meta

    def __add_labels_file(self, meta):
        # Adds the labels to the metadata
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.labels_path)
        label_file.description = "Labels for objects that the model can detect."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
        meta.associatedFiles = [label_file]
