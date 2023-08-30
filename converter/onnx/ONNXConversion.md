# Conversion

## YOLOv5 (PyTorch) to ONNX format conversion

**Supported models**: The conversion is supported for models of Yolov5 and Yolov8 with all sizes.
The detection and segmentation Yolov5 and Yolov8 models are both supported; the classification or pose models are not.

**Supported types**: The converter supports the 3 types, float32, float16 int8.

**Supported input resolutions**: Any size that is a multiple of 32.
Only square inputs are considered at the moment, i.e. width=height.

**Other**: Both NMS and normalization are automatically added to the converted model.

**Limitations**:
- Having the IoU and confidence as inputs to the model is not supported (e.g. they will be hard-coded into the model).

### Usage

Use:

```bash
python onnx/convert.py \
    --model path/to/model.pt \
    --input-resolution 640 \
    --quantize-model float32 float16 int8 \
    --iou-threshold 0.45 \
    --conf-threshold 0.25
    
```
- The conversion script takes the PyTorch model as input (`--model`), the directory in which to save the converted model, as well as its name (`--out`and `--output-name`), i.e. the line above will produce models named `path/to/model_640_float32.onnx` as well as the models with float16 and int8.
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
import ast
import onnx

model = onnx.load("path/to/model.onnx")
metadata = model.metadata_props
labels = [ast.literal_eval(m.value) for m in metadata if m.key == 'names'][0]
```