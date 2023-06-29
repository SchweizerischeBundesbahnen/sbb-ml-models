# Conversion

## YOLOv5 (PyTorch) to ONNX format conversion

**Supported models**: The conversion is supported for models of Yolov5 v4.0 and v5.0 with size S, S6, M, M6, L, L6 and X, X6, as well as N, N6 for v6.0.
The detection Yolov5 models is supported; the classification and segmentation model are not.

**Supported types**: The converter supports the 3 types, float32, float16 int8.

**Supported input resolutions**: Any size that is a multiple of 32.
Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are automatically added to the converted model.

**Limitations**:
- Having the IoU and confidence as inputs to the model is not supported (e.g. they will be hard-coded into the model).

### Usage
Go to the onnx directory and use:
```bash
poetry run python convert.py \
    --model PATH_TO_PT_MODEL \
    --out PATH_TO_OUTPUT_DIR \
    --output-name CONVERTED_MODEL_NAME \
    --input-resolution INPUT_RESOLUTION \
    --quantize-model float32 int8
```
- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce models named `CONVERTED_MODEL_NAME.onnx`.
- By default, the input resolution is 640 and the model is converted only in float32.
- The model does not support thresholds as inputs. Thus, the input to the model is only an image with 0-255 values.
- The output contains the bounding box coordinates and the class and score of each box.
- The thresholds used for NMS can be specified using `--iou-threshold` and `--conf-threshold`, they will be hard-coded in the model.

### YoloV5 ONNX Model for Detection

Input:

* **image**: an image array (1 x INPUT_RESOLUTION x INPUT_RESOLUTION x 3) (required)

Output:

* **location**: The coordinates for each detected box (1 x NB_DETECTION x [top, left, bottom, right]). The coordinates
  are relative to the image size.
* **category**:  The category id for each detected box (1 x NB_DETECTION)
* **score**: The confidence score for each detected box (1 x NB_DETECTION)

Labels:
* They are included in the converted model, as metadata.

```python
import onnx

model = onnx.load("path_to_the_onnx_model")
metadata = model.metadata_props
labels = metadata[0].value.split(',')
```