from constants import DEFAULT_INPUT_RESOLUTION, DEFAULT_SOURCE_DATASET, DEFAULT_NB_CALIBRATION, \
    DEFAULT_MAX_NUMBER_DETECTION, FLOAT32, FLOAT16, INT8, \
    FULLINT8, BAHNHOF, WAGEN, TRAKTION, TFLITE, SAVED_MODEL, PADDED, SIMPLE, COMBINED


class ModelParameters:
    '''
    Parameters used for the model
    '''

    def __init__(self, img_size=(DEFAULT_INPUT_RESOLUTION, DEFAULT_INPUT_RESOLUTION),
                 include_nms=True,
                 include_normalization=True,
                 include_threshold=False,
                 tf_raw_resize=False,
                 nb_classes=-1):
        img_size = img_size * 2 if len(img_size) == 1 else img_size  # expand
        self.img_size = img_size
        self.include_nms = include_nms
        self.include_normalization = include_normalization
        self.include_threshold = include_threshold
        self.tf_raw_resize = tf_raw_resize
        self.nb_classes = nb_classes


class ConversionParameters:
    '''
    Parameters for the conversion
    '''

    def __init__(self, dest=TFLITE,
                 quantization_types=None,
                 source=DEFAULT_SOURCE_DATASET,
                 nb_calib=DEFAULT_NB_CALIBRATION):
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

        if source not in [BAHNHOF, WAGEN, TRAKTION]:
            raise ValueError(f"Source '{source}' not recognized: must be one of '{BAHNHOF}', '{WAGEN}', '{TRAKTION}'.")
        self.source = source
        self.nb_calib = nb_calib


class PostprocessingParameters:
    '''
    Parameters for the postprocessing
    '''

    def __init__(self, max_det=DEFAULT_MAX_NUMBER_DETECTION,
                 nms_type=PADDED):
        self.max_det = max_det
        if nms_type not in [SIMPLE, PADDED, COMBINED]:
            raise ValueError(
                f"NMS algorithm '{nms_type}' not recognized: must be one of '{SIMPLE}', '{PADDED}', '{COMBINED}'.")
        self.nms_type = nms_type
