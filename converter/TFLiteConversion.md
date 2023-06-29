# Conversion

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