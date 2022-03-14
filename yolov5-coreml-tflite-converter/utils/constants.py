import os

# -------------------------------------------------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------------------------------------------------- #
# General
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
DATA_DIR = os.path.join('data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

FLOAT32 = 'float32'
FLOAT16 = 'float16'
INT8 = 'int8'
FULLINT8 = 'fullint8'
FLOAT32_SUFFIX = '_float32'
FLOAT16_SUFFIX = '_float16'
INT8_SUFFIX = '_int8'
FULLINT8_SUFFIX = '_fullint8'

BATCH_SIZE = 1
NB_CHANNEL = 3

# x, y, w, h, score, class1, class2, ...
XY_SLICE = (0, 2)
WH_SLICE = (2, 4)
SCORE_SLICE = (4, 5)
CLASSES_SLICE = (5, 0)
NB_OUTPUTS = 5  # 1 objectness score + 4 bounding box coordinates
NORMALIZATION_FACTOR = 255.

# Input names
IMAGE_NAME = 'image'
NORMALIZED_SUFFIX = '_normalized'
QUANTIZED_SUFFIX = '_quantized'
IOU_NAME = 'iou threshold'
CONF_NAME = 'conf threshold'

# Colors
BLUE = '\033[36m'
GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
PURPLE = '\033[34m'
END_COLOR = '\033[0m'
BOLD = '\033[1m'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# CoreML converter
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
COREML_SUFFIX = '.mlmodel'
TORCHSCRIPT_SUFFIX = '.torchscript.pt'

# Outputs names
CONFIDENCE_NAME = 'confidence'  # list of class scores
COORDINATES_NAME = 'coordinates'  # (x, y, w, h)
RAW_PREFIX = 'raw_'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# TFLite converter
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
TFLITE_SUFFIX = '.tflite'
LABELS_NAME = 'labels.txt'

# Format
TFLITE = 'tflite'
SAVED_MODEL = 'saved_model'
GRAPH_DEF_SUFFIX = '.pb'

# NMS
PADDED = 'padded'
SIMPLE = 'simple'
COMBINED = 'combined'

# Representative dataset
BAHNHOF = 'bahnhof'
WAGEN = 'wagen'
TRAKTION = 'traktion'

# Output names
BOUNDINGBOX_NAME = 'location'  # (y1, x1, y2, x2)
CLASSES_NAME = 'category'  # class index
SCORES_NAME = 'score'  # confidence score
NUMBER_NAME = 'number of detections'  # number of detected object in the image
DETECTIONS_NAME = 'detection results'
PREDICTIONS_NAME = 'predictions'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# ONNX converter
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
ONNX_SUFFIX = '.onnx'
OPSET = 12

# -------------------------------------------------------------------------------------------------------------------- #
# Default values
# -------------------------------------------------------------------------------------------------------------------- #
DEFAULT_COREML_NAME = 'yolov5-coreML'
DEFAULT_TFLITE_NAME = 'yolov5-TFLite'
DEFAULT_ONNX_NAME = 'yolov5-ONNX'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# Common values
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
DEFAULT_MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'converted_models')
DEFAULT_PT_MODEL = os.path.join('data', 'models', 'best.pt')
DEFAULT_INPUT_RESOLUTION = 640
DEFAULT_QUANTIZATION_TYPE = FLOAT32
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_CONF_THRESHOLD = 0.25

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# TFlite additional default values
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
DEFAULT_SOURCE_DATASET = WAGEN
DEFAULT_NB_CALIBRATION = 500
DEFAULT_MAX_NUMBER_DETECTION = 20


def get_zipfile_path(source):
    return os.path.join(DATA_DIR, f'{source}_500.zip')


def get_dataset_url(source):
    return f'https://sbb-ml-public-resources-prod.s3.eu-central-1.amazonaws.com/quantization/{source}_500.zip'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# Inference
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
DEFAULT_DETECTED_IMAGE_DIR = os.path.join(OUTPUT_DIR, 'detections')
