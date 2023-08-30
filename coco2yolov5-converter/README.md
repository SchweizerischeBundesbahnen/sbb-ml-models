# Coco to Yolo format conversion
This code is used to convert a dataset in [COCO format](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch#:~:text=The%20COCO%20dataset%20is%20formatted,%E2%80%9D%20(in%20one%20case).) to a dataset in [Yolo format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) which can be used to train a YoloV5 model.

## Setup
Python 3.8 is used. Install the dependencies with pip: `pip3 install -r requirements.txt`

## Usage
Use:
 ```
 python src/sbb2yolo.py --coco-input-folders PATH_TO_COCO_INPUT_FOLDER1 PATH_TO_COCO_INPUT_FOLDER2 \
  --yolo-output-folder PATH_TO_OUTPUT_FOLDER \
  --yolo-config-file CONFIG_FILE_NAME \
  --dataset-split-pivot RATIO_TRAIN_TO_VAL
 ```
- The conversion script takes the Coco input folders (folder containing all images as well as an annotation file). Several Coco datasets can be specified, in which case the script combines all the datasets.
- It creates a datast in Yolo format in `PATH_TO_OUTPUT_FOLDER`. This dataset contains the configuration file, as well as an `images` and `labels` folder 
- The name of the YAML config file can be changed (`--yolo-config-file`)
- The proportion of the data to use for test is specified with `--dataset-split-pivot`, e.g. 0.8 means 80% for the training set and the remaining 20% for the validation set.

 ## Tests
In order to run the tests, use:
 ```
 pytest
 ```
 
## Authors
* **Stefan Aebischer**
* **Jeanne Fleury**
