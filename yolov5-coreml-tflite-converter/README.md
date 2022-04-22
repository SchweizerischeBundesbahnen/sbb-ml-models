# Conversion
## Setup

The code requires the original YOLOv5 code, as found here: https://github.com/ultralytics/yolov5/tree/v6.0. 

In order for the dependency install to work, you need to:
- provide a copy of that repository at `yolov5`, relative to this file (`git clone -b v6.0 --depth 1 https://github.com/ultralytics/yolov5`)
- add a `setup.py` file in that directory. The minimal content for that file can be taken from `yolov5.setup.py`
- replace the two files `yolov5/models/yolo.py` and `yolov5/models/tf.py` by `yolov5.yolo.py` and `yolov5.tf.py`, renamed to `yolo.py` and `tf.py`, respectively

After this is done, install the dependencies with pip: `pip install -r requirements.txt`.


## YOLOv5 (PyTorch) to CoreML format conversion

This code was taken originally from https://github.com/dbsystel/yolov5-coreml-tools. It was released there under the Apache 2.0 license (see coreml/LICENSE file). In accordance with the license, the code in this repository has been modified, but the original comments about the license are still present. Where applicable, new comments show that these files have been modified after their release under the original license. The modified code **DOES NOT** fall under the Apache 2.0 license.

Also refer to coreml/README-original.md for further information about the original source code by DB Systel GmbH.

**Supported models**: The conversion is supported for models of Yolov5 v4.0, v5.0 and v6.0 with size S, S6, M, M6, L, L6 and X, X6, as well as N, N6 for v6.0.

**Supported types**: The converter supports the 3 types, float32, float16 and int8.

**Supported input resolutions**: Any size that is a multiple of 32. Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are automatically added to the converted model.

### Usage
Use:
```bash
python coreml/convert.py \
    --model PATH_TO_PT_MODEL \
    --out PATH_TO_OUTPUT_DIR \
    --output-name CONVERTED_MODEL_NAME \
    --input-resolution INPUT_RESOLUTION \
    --quantize-model float32 float16 int8
```

- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce the model `PATH_TO_OUTPUT_DIR/CONVERTED_MODEL_NAME.mlmodel`.
- By default, the input resolution is 640 and the model is converted only in float32.
- NMS and normalization are included into the converted model. Thus, the input to the model is an image with [0-255] values as well as the IoU and confidence thresholds. The output contains the bounding box coordinates and the confidence for each classes.

### YoloV5 CoreML Model
Input: 
* **image**: an image (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iou threshold**: the IoU threshold used during NMS (Default: 0.45)
* **confidence threshold**: the confidence threshold used during NMS (Default: 0.25)

Output:
* **confidence**: The confidence of each category, for each detected box (NB_DETECTIONS x NB_CATEGORIES)
* **coordinates**: The coordinates for each detected box (NB_DETECTIONS x [x center, y center, width, height]). The coordinates are relative to the image size.


## YOLOv5 (PyTorch) to TFLite format conversion

The original code is taken from https://github.com/zldrobit/yolov5/blob/tf-android/models/tf.py and modified to suit our purpose.

**Supported models**: The conversion is supported for models of Yolov5 v4.0 and v5.0 with size S, S6, M, M6 and L as well as N, N6 for v6.0. Larger sizes L6, X and X6 are currently subject to some limitations.

**Supported types**: The converter supports the 3 types, float32, float16 and int8.

**Supported input resolutions**: Any size that is a multiple of 32. Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are added by default to the converted model, but both can also be removed if wanted.

**Limitations**:
- The larger models L6, X, X6 can be converted using float32.
- They are not compatible with quantization float16 and int8, if NMS is included in the model.
- A workaround is to use Combined NMS (`--nms-type combined`), which then requires that the TFLite interpreter needs to link Flex delegate in order to run the model since it contains some flex ops.

### Usage
Use:
```bash
python tflite/convert.py \
    --model PATH_TO_PT_MODEL \
    --out PATH_TO_OUTPUT_DIR \
    --output-name CONVERTED_MODEL_NAME \
    --input-resolution INPUT_RESOLUTION \
    --quantize-model float32 float16 int8
```

- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce the model `PATH_TO_OUTPUT_DIR/CONVERTED_MODEL_NAME.tflite`.
- By default, the input resolution is 640 and the model is converted only in float32.
- By default, both normalization and NMS are added, thus the model expects an image with values [0-255], as well as the IoU and confidence thresholds as input. The output contains the bounding boxes, the predicted class for each box as well as its score, and the number of detected objects.
- The normalization can be removed, in which case the model will expect a normalized image [0-1] instead of [0-255].
- NMS can also be removed, in which case the model returns the list of eligible predictions (to be filtered)

### YoloV5 TFLite Model
#### NMS included
Input: 
* **image**: an image (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iou threshold**: the IoU threshold used during NMS (required)
* **confidence threshold**: the confidence threshold used during NMS (required)

Output:
* **location**: The coordinates for each detected box (1 x NB_DETECTION x [top, left, bottom, right]). The coordinates are relative to the image size.
* **category**:  The category id for each detected box (1 x NB_DETECTION)
* **score**: The confidence score for each detected box (1 x NB_DETECTION)
* **number of detections**: The number of detected box (NB_DETECTION)

#### NMS not included
Input: 
* **image**: an image (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)

Output:
* **predictions**: The predictions made by YoloV5 (1 x NB_PREDICTIONS x [x, y, w, h, confidence, class1, ..., classX]). The coordinates are absolute.

# Inference with Python
This code demonstrates inference using a converted YOLOv5 model using Python.
Use:
```bash
python inference-python/detect.py --model PATH_TO_CONVERTED_MODEL \
--img-dir PATH_TO_IMG_DIR \
--max-img MAX_IMG \
--out OUT_PATH \
--iou_threshold IOU_THRESHOLD \
--conf-threshold CONF_THRESHOLD \
--no-save
```
The path to the model (PyTorch, CoreML or TFLite) must be specified with `--model`. The path to the directory containing the images to test the model on with is specified `--img-dir`. If the repository contains many images and one wants to only test on a few, the maximum number of images to use can be given with `--max-img`. By default, the detections are saved in `output/data/detections`, that path can be changed with `--out`. The IoU and confidence thresholds can be given with `--iou-threshold` and `--conf-threshold`, respectively. Additonally, if one doesn't want to save the detections, one can use the flag `--no-save`.
# Test
Once converted, the model can be tested.

Use:
```bash
python inference-python/test_model.py --model PATH_TO_CONVERTED_MODEL \
--reference-model PATH_TO_REFERENCE_MODEL \
--img-dir PATH_TO_IMG_DIR \
--verbose
```
In order to test a model, one needs a reference model. The detections made by the compared model are then compared to those of the reference model, typically one uses the PyTorch model as reference. The path to the compared model is given with `--model`, and the path to the reference model with `--reference-model`. Any model (PyTorch, CoreML, TFLite) can be used, but both models must consider the same classes. The models are compared on a set of images given by `--img-dir`. The flag `--verbose` can be used if one wants the details of each predictions. The test passes if the mAP score is above 0.6 and fails otherwise.

## Authors
* **Jeanne Fleury**
