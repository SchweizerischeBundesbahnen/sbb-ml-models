# Conversion

## Setup

The code requires the original YOLOv5 code, as found here: https://github.com/ultralytics/yolov5/tree/v6.0.

Pipenv is used to handle the dependencies, so make sure you have Pipenv installed on your machine.
Then run `./install.sh`. It will pull the code from the Yolov5 repo, and install all dependencies.
Run `pipenv shell` to activate the pipenv environment and you're ready.

On Macs M1, conda is used to handle the dependencies.
Use `./init_conda_m1.sh` to create the conda environment and pull the code from the Yolov5 repo.
Activate the environment with `conda activate converter_yolo` and install the dependencies
with `./install_dependencies_m1.sh`.

There may be some issue with the torch library, and workarounds may be found
here: https://github.com/ultralytics/yolov5/issues/6948#issuecomment-1075528897

## YOLOv5 (PyTorch) to CoreML format conversion

This code was taken originally from https://github.com/dbsystel/yolov5-coreml-tools.
It was released there under the Apache 2.0 license (see coreml/LICENSE file).
In accordance with the license, the code in this repository has been modified,
but the original comments about the license are still present. Where applicable,
new comments show that these files have been modified after their release under the original license.
The modified code **DOES NOT** fall under the Apache 2.0 license.

Also refer to coreml/README-original.md for further information about the original source code by DB Systel GmbH.

**Supported models**: The conversion is supported for models of Yolov5 v4.0, v5.0, v6.0, v6.1, v6.2 and v7.0 with all
sizes.
The detection and segmentation Yolov5 models are both supported; the classification model is not.

**Supported types**: The converter supports the 3 types, float32, float16 and int8.

**Supported input resolutions**: Any size that is a multiple of 32.
Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are automatically added to the converted model.
For the segmentation model, masks calculations are also included in the model

**Note**: The postprocessing for the detection model is built-in,
that is the inputs for the thresholds are double,
there is a `preview' tab in the model to easily run the inference on images.
However, for the segmentation model,
more flexibility was needed in order to compute the masks,
and so the input thresholds are arrays (e.g. [0.5] instead of only 0.5) and there is no 'preview' tab.
Furthermore, the number of detected boxes is not flexible, but given by a parameter, which can be chosen.
The model then additionally returns the number of detections actually made.

### Usage

Use:

```bash
python coreml/convert.py \
    --model PATH_TO_PT_MODEL \
    --out PATH_TO_OUTPUT_DIR \
    --output-name CONVERTED_MODEL_NAME \
    --input-resolution INPUT_RESOLUTION \
    --quantize-model float32 float16 int8 \
    --max-det MAX_DET
```

- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted
  model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce the
  model `PATH_TO_OUTPUT_DIR/CONVERTED_MODEL_NAME.mlmodel`.
- By default, the input resolution is 640 and the model is converted only in float32.
- NMS and normalization are included into the converted model. Thus, the input to the model is an image with [0-255]
  values as well as the IoU and confidence thresholds. The output contains the bounding box coordinates and the
  confidence for each classes. For the segmentation model, the masks are also returned.
- `--max-det` is the maximum number of detection made by the model, only used for the segmentation model.

### YoloV5 CoreML Model for Detection

Input:

* **image**: an image, the input image which must be square (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iou threshold**: a double, the IoU threshold used during NMS (Default: 0.45)
* **confidence threshold**: a double, the confidence threshold used during NMS (Default: 0.25)

Output:

* **confidence**: The confidence of each category, for each detected box (NB_DETECTIONS x NB_CATEGORIES)
* **coordinates**: The coordinates for each detected box (NB_DETECTIONS x [x center, y center, width, height]). The
  coordinates are relative to the image size.

Labels:

* For the detection model, the labels are contained in the model (e.g. we used built-in NMS function),
  they can be found in the specifications of the model:

```python
from coremltools.models.model import MLModel

coreml_model = model = MLModel("path_to_the_coreml_model")
spec = coreml_model.get_spec()
labels = [label.rstrip() for label in spec.pipeline.models[-1].nonMaximumSuppression.stringClassLabels.vector]
```

### YoloV5 CoreML Model for Segmentation

Input:

* **image**: an image, which must be square (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iou threshold**: an array, the IoU threshold used during NMS (Default: [0.45])
* **confidence threshold**: an array, the confidence threshold used during NMS (Default: [0.25])

Output:

* **confidence**: The confidence of each category, for each detected box (NB_DETECTIONS x NB_CATEGORIES)
* **coordinates**: The coordinates for each detected box (NB_DETECTIONS x [x center, y center, width, height]). The
  coordinates are relative to the image size.
* **masks**: The masks for the segmentation for each detected box (NB_DETECTIONS x INPUT_RESOLUTION x INPUT_RESOLUTION)
* **number of detections**: The number of detected boxes

Labels:

* For the segmentation model, the labels are also contained in the model.
  At the moment, they are in the description of the masks, as we have not yet figured out how to include them in a
  proper way.

```python
from coremltools.models.model import MLModel

coreml_model = model = MLModel("path_to_the_coreml_model")
spec = coreml_model.get_spec()
labels = spec.description.output[-1].shortDescription.split(',')
```

## YOLOv5 (PyTorch) to TFLite format conversion

The original code is taken from https://github.com/zldrobit/yolov5/blob/tf-android/models/tf.py and modified to suit our
purpose.

**Supported models**: The conversion is supported for models of Yolov5 v4.0, v5.0, v6.0, v6.1, v6.2 and v7.0 with all
sizes.
The detection and segmentation Yolov5 models are both supported; the classification model is not.

**Supported types**: The converter supports the 3 types, float32, float16 and int8.

**Supported input resolutions**: Any size that is a multiple of 32.
Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are automatically added to the converted model.
For the segmentation model, masks calculations are also included in the model

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

- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted
  model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce the
  model `PATH_TO_OUTPUT_DIR/CONVERTED_MODEL_NAME.tflite`.
- By default, the input resolution is 640 and the model is converted only in float32.
- By default, both normalization and NMS are added, thus the model expects an image with values [0-255], as well as the
  IoU and confidence thresholds as input. The output contains the bounding boxes, the predicted class for each box as
  well as its score, and the number of detected objects.
- The normalization can be removed, in which case the model will expect a normalized image [0-1] instead of [0-255].
- NMS can also be removed, in which case the model returns the list of eligible predictions (to be filtered)

### YoloV5 TFLite Model for Detection

Input:

* **image**: an image array (1 x INPUT_RESOLUTION x INPUT_RESOLUTION x 3) (required)
* **iou threshold**: the IoU threshold used during NMS (1,) (required)
* **confidence threshold**: the confidence threshold used during NMS (1,) (required)

Output:

* **location**: The coordinates for each detected box (1 x NB_DETECTION x [top, left, bottom, right]). The coordinates
  are relative to the image size.
* **category**:  The category id for each detected box (1 x NB_DETECTION)
* **score**: The confidence score for each detected box (1 x NB_DETECTION)
* **number of detections**: The number of detected box (1,)

### YoloV5 TFLite Model for Segmentation

Input:

* **image**: an image array (1 x INPUT_RESOLUTION x INPUT_RESOLUTION x 3) (required)
* **iou threshold**: the IoU threshold used during NMS (1,) (required)
* **confidence threshold**: the confidence threshold used during NMS (1,) (required)

Output:

* **location**: The coordinates for each detected box (1 x NB_DETECTION x [top, left, bottom, right]). The coordinates
  are relative to the image size.
* **category**:  The category id for each detected box (1 x NB_DETECTION)
* **score**: The confidence score for each detected box (1 x NB_DETECTION)
* **masks**: The mask for each detected box (NB_DETECTION x INPUT_RESOLUTION x INPUT_RESOLUTION)
* **number of detections**: The number of detected box (1,)

### Associated data

Labels:

* They are included in the converted model, the model can be unzipped, and you'll then get the `labels.txt`. They can
  also be retrieved with the metadata.

```python
from tflite_support import metadata

metadata_displayer = metadata.MetadataDisplayer.with_model_file("path_to_the_tflite_model")
labels = metadata_displayer.get_associated_file_buffer('labels.txt').decode().split('\n')[:-1]
```

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

The path to the model (PyTorch, CoreML or TFLite) must be specified with `--model`.
The path to the directory containing the images to test the model on with is specified `--img-dir`.
If the repository contains many images and one wants to only test on a few,
the maximum number of images to use can be given with `--max-img`.
By default, the detections are saved in `output/data/detections`,
that path can be changed with `--out`. The IoU and confidence thresholds can be given with `--iou-threshold`
and `--conf-threshold`, respectively.
Additonally, if one doesn't want to save the detections, one can use the flag `--no-save`.

# Test

Once converted, the model can be tested.

Use:

```bash
python inference-python/test_model.py --model PATH_TO_CONVERTED_MODEL \
--reference-model PATH_TO_REFERENCE_MODEL \
--img-dir PATH_TO_IMG_DIR \
--verbose
```

In order to test a model, one needs a reference model.
The detections made by the compared model are then compared to those of the reference model,
typically one uses the PyTorch model as reference. The path to the compared model is given with `--model`,
and the path to the reference model with `--reference-model`.
Any model (PyTorch, CoreML, TFLite) can be used, but both models must consider the same classes.
The models are compared on a set of images given by `--img-dir`.
The flag `--verbose` can be used if one wants the details of each predictions.
The test passes if the mAP score is above 0.6 and fails otherwise.

## Authors

* **Jeanne Fleury**
