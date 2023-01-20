from typing import List

from helpers.constants import DEFAULT_INPUT_RESOLUTION, DEFAULT_SOURCE_DATASET, DEFAULT_NB_CALIBRATION, \
    DEFAULT_MAX_NUMBER_DETECTION, FLOAT32, FLOAT16, INT8, \
    FULLINT8, BAHNHOF, WAGEN, TRAKTION, TFLITE, SAVED_MODEL, PADDED, SIMPLE, COMBINED, DETECTION, SEGMENTATION


class ModelParameters:
    """ Parameters used for the model

    Attributes
    ----------
    model_type: str
        The model type, either `detection` or `segmentation`

    img_size: (int, int)
        The input size

    include_nms: bool
        Whether to include NMS in the model

    include_normalization: bool
        Whether to include normalization in the model

    include_threshold: bool
        Whether to include the thresholds (iou, conf) in the model - if nms is included

    nb_classes: int
        The number of classes detected by the model
    """
    def __init__(self, model_type: str = DETECTION,
                 img_size: (int, int) = (DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION),
                 include_nms: bool = True,
                 include_normalization: bool = True,
                 include_threshold: bool = False,
                 nb_classes: int = -1):
        self.model_type = model_type
        if model_type not in [DETECTION, SEGMENTATION]:
            raise ValueError(
                f"The model type: '{model_type}' is not supported. Needs to bo one of '{DETECTION}', '{SEGMENTATION}'.")

        img_size = img_size * 2 if len(img_size) == 1 else img_size  # expand
        self.img_size = img_size
        self.include_nms = include_nms
        self.include_normalization = include_normalization
        self.include_threshold = include_threshold
        self.nb_classes = nb_classes


class ConversionParameters:
    """ Parameters for the conversion

    Attributes
    ----------
    dest: str
        The destination, e.g. TFlite or saved_model

    quantization_types: List[str]
        The quantization types to use for the conversion

    write_metadata: bool
        Whether to include metadata in the model

    use_representative_dataset: bool
        Whether to use representative dataset when doing quantization

    source: str
        The source for the representative dataset if any (`bahnhof`, `wagen`, `traktion`)
    """
    def __init__(self, dest: str = TFLITE,
                 quantization_types: List[str] = None,
                 write_metadata: bool = True,
                 use_representative_dataset: bool = False,
                 source: str = DEFAULT_SOURCE_DATASET,
                 nb_calib: int = DEFAULT_NB_CALIBRATION):
        if dest not in [SAVED_MODEL, TFLITE]:
            raise ValueError(
                f"Destination format '{dest}'' not recognized: must be one of '{SAVED_MODEL}', '{TFLITE}'.")
        self.dest = dest

        if quantization_types is None:
            quantization_types = [FLOAT32]
        for quantization_type in quantization_types:
            if quantization_type not in [FLOAT32, FLOAT16, INT8, FULLINT8]:
                raise ValueError(
                    f"Quantization option '{quantization_type}' not recognized: must be one of '{FLOAT32}', '{FLOAT16}', '{INT8}', '{FULLINT8}'.")
        self.quantization_types = quantization_types
        self.write_metadata = write_metadata
        self.use_representative_dataset = use_representative_dataset
        if source not in [BAHNHOF, WAGEN, TRAKTION]:
            raise ValueError(f"Source '{source}' not recognized: must be one of '{BAHNHOF}', '{WAGEN}', '{TRAKTION}'.")
        self.source = source
        self.nb_calib = nb_calib


class PostprocessingParameters:
    """ Parameters for the postprocessing

    Attributes
    ----------
    max_det: int
        The max number of detections made by the converted model

    nms_type: str
        The type of NMS to use if NMS is included (`simple`, `padded`, `combined`)
    """
    def __init__(self, max_det: int = DEFAULT_MAX_NUMBER_DETECTION,
                 nms_type: str = PADDED):
        self.max_det = max_det
        if nms_type not in [SIMPLE, PADDED, COMBINED]:
            raise ValueError(
                f"NMS algorithm '{nms_type}' not recognized: must be one of '{SIMPLE}', '{PADDED}', '{COMBINED}'.")
        self.nms_type = nms_type
