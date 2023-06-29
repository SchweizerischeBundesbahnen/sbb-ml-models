# Conversion

## Setup

The code requires the original YOLOv5 code, as found here: https://github.com/ultralytics/yolov5/tree/v6.0.

Pipenv is used to handle the dependencies, so make sure you have Pipenv installed on your machine.
Then run `./install.sh`. It will pull the code from the Yolov5 repo, and install all dependencies.
Run `pipenv shell` to activate the pipenv environment and you're ready.

On Macs M1, conda is used to handle the dependencies (https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), 
bazel 5.1.1 is also required (https://bazel.build/install/os-x).
If you have e.g. bazel 6.0.0, run `cd "/opt/homebrew/Cellar/bazel/6.0.0/libexec/bin" && curl -fLO https://releases.bazel.build/5.1.1/release/bazel-5.1.1-darwin-arm64 && chmod +x bazel-5.1.1-darwin-arm64` to change to 5.1.1.
Use `./init_conda_m1.sh` to create the conda environment and pull the code from the Yolov5 repo.
Activate the environment with `conda activate converter_yolo` and install the dependencies
with `./install_dependencies_m1.sh`.

There may be some issue with the torch library, and workarounds may be found
here: https://github.com/ultralytics/yolov5/issues/6948#issuecomment-1075528897
the maximum number of images to use can be given with `--max-img`.
By default, the detections are saved in `output/data/detections`,
that path can be changed with `--out`. The IoU and confidence thresholds can be given with `--iou-threshold`
and `--conf-threshold`, respectively.
Additonally, if one doesn't want to save the detections, one can use the flag `--no-save`.

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