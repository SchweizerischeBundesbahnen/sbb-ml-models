from typing import List

from helpers.constants import DEFAULT_INPUT_RESOLUTION, DEFAULT_SOURCE_DATASET, DEFAULT_NB_CALIBRATION, \
    DEFAULT_MAX_NUMBER_DETECTION, FLOAT32, FLOAT16, INT8, \
    FULLINT8, BAHNHOF, WAGEN, TRAKTION, DETECTION, SEGMENTATION, UNKNOWN


class ModelParameters:
    """ Parameters used for the model

    Attributes
    ----------
    model_type: str
        The model type, either `detection` or `segmentation`

    input_resolution: int
        The input resolution

    include_nms: bool
        Whether to include NMS in the model

    include_normalization: bool
        Whether to include normalization in the model

    max_det: int
        The maximum number of detections

    nb_classes: int
        The number of classes detected by the model
    """

    def __init__(self, model_type: str = UNKNOWN,
                 model_orig: str = UNKNOWN,
                 input_resolution: int = DEFAULT_INPUT_RESOLUTION,
                 include_nms: bool = True,
                 include_normalization: bool = True,
                 max_det: int = DEFAULT_MAX_NUMBER_DETECTION,
                 nb_classes: int = -1):
        self.model_type = model_type
        if model_type not in [UNKNOWN, DETECTION, SEGMENTATION]:
            raise ValueError(
                f"The model type: '{model_type}' is not supported. Needs to bo one of '{DETECTION}', '{SEGMENTATION}'.")
        self.model_orig = model_orig
        self.input_resolution = input_resolution
        self.include_nms = include_nms
        self.include_normalization = include_normalization
        self.max_det = max_det
        self.nb_classes = nb_classes


class ConversionParameters:
    """ Parameters for the conversion

    Attributes
    ----------
    quantization_types: List[str]
        The quantization types to use for the conversion

    write_metadata: bool
        Whether to include metadata in the model

    use_representative_dataset: bool
        Whether to use representative dataset when doing quantization

    source: str
        The source for the representative dataset if any (`bahnhof`, `wagen`, `traktion`)
    """

    def __init__(self, quantization_types: List[str] = None,
                 write_metadata: bool = True,
                 use_representative_dataset: bool = False,
                 source: str = DEFAULT_SOURCE_DATASET,
                 nb_calib: int = DEFAULT_NB_CALIBRATION):
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
