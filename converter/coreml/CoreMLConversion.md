# CoreML conversion

## YOLOv5 (PyTorch) to CoreML format conversion

This code was taken originally from https://github.com/dbsystel/yolov5-coreml-tools.
It was released there under the Apache 2.0 license (see coreml/LICENSE file).
In accordance with the license, the code in this repository has been modified,
but the original comments about the license are still present. Where applicable,
new comments show that these files have been modified after their release under the original license.
The modified code **DOES NOT** fall under the Apache 2.0 license.

Also refer to coreml/README-original.md for further information about the original source code by DB Systel GmbH.

**Supported models**: The conversion is supported for models of Yolov5 and Yolov8 with all sizes.
The detection and segmentation Yolov5 and Yolov8 models are both supported; the classification or pose models are not.

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
The model then additionally returns the number of detections actually made, with the maximum being the number of boxes passed as parameters.

### Usage

Use:

```bash
python coreml/convert.py \
    --model path/to/model.pt \
    --input-resolution 640 \
    --quantize-model float32 float16 int8 \
    --max-det MAX_DET \
    --overwrite
```

- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted
  model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce the
  model `path/to/model_640_float32.mlmodel` as well as the ones with float16 and int8.
- By default, the input resolution is 640 and the model is converted only in float32.
- NMS and normalization are included into the converted model. Thus, the input to the model is an image with [0-255]
  values as well as the IoU and confidence thresholds. The output contains the bounding box coordinates and the
  confidence for each classes. For the segmentation model, the masks are also returned.
- `--max-det` is the maximum number of detection made by the model, only used for the segmentation model.

### YoloV5 CoreML Model for Detection

Input:

* **image**: an image, the input image which must be square (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iouThreshold**: a double, the IoU threshold used during NMS (Default: 0.45)
* **confidenceThreshold**: a double, the confidence threshold used during NMS (Default: 0.25)

Output:

* **confidence**: The confidence of each category, for each detected box (NB_DETECTIONS x NB_CATEGORIES)
* **coordinates**: The coordinates for each detected box (NB_DETECTIONS x [x center, y center, width, height]). The
  coordinates are relative to the image size.

### YoloV5 CoreML Model for Segmentation

Input:

* **image**: an image, which must be square (INPUT_RESOLUTION x INPUT_RESOLUTION) (required)
* **iouThreshold**: an array, the IoU threshold used during NMS (Default: [0.45])
* **confidenceThreshold**: an array, the confidence threshold used during NMS (Default: [0.25])

Output:

* **confidence**: The confidence of each category, for each detected box (NB_DETECTIONS x NB_CATEGORIES)
* **coordinates**: The coordinates for each detected box (NB_DETECTIONS x [x center, y center, width, height]). The
  coordinates are relative to the image size.
* **masks**: The masks for the segmentation for each detected box (NB_DETECTIONS x INPUT_RESOLUTION x INPUT_RESOLUTION)
* **numberDetections**: The number of detected boxes

Labels:

* The labels can be found in the metadata of the model.

```python
import ast
from coremltools.models.model import MLModel

model = MLModel("path/to/model.mlmodel")
metadata = model.user_defined_metadata
labels = ast.literal_eval(metadata['names'])

```
