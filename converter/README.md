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

# Conversion Yolov5 (Pytorch)

This repo supports the conversion of the Yolov5 model (PyTorch) to CoreML, TFLite and ONNX, with some restrictions.
- [CoreML Conversion](coreml/CoreMLConversion.md)
- [TFLIte Conversion](tflite/TFLiteConversion.md)
- [ONNX Conversion](onnx/ONNXConversion.md)

# Inference with Python

This code demonstrates inference using a converted YOLOv5 model using Python.
Use:

```bash
python inference/detect.py --model PATH_TO_CONVERTED_MODEL \
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
python inference/compare_model.py --model PATH_TO_CONVERTED_MODEL \
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
