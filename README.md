# SBB ML Models

## Introduction
This repository contains trained YoloV5 models, as well as all tools necessary in order to have YoloV5 run on mobile.

<!---
What's the purpose of this repository ?
-->

## Model overview
We have several models which consider different datasets, i.e. recognize different objects:
- **Bahnhof**: this model recognizes the object typically found in a train station
- **Wagen**: this model recognizes the objects inside a train
- **Traktion**: this model is more specific, it handles all objects which have to do with locomotives
- **Universal** (all of the above)

The description of each label can be found [here](Annotations.md).

<!---
## Demo
Include some demo videos
-->

## How to use the models
In order for a YoloV5 model to be used on mobile, one needs to convert it to a suitable format. This is CoreML for an iOS app and TFLite for an Android app.
All information required to convert a model can be found [here](yolov5-coreml-tflite-converter/README.md).

We have also developed libraries in order to facilitate the deployment of the converted YoloV5 model on mobile:
[Android library](https://github.com/SchweizerischeBundesbahnen/mobile-android-ml) and [iOS library](https://github.com/SchweizerischeBundesbahnen/mobile-ios-ml)

## Additional information
The Coco dataset format is a widely-used format for object detection. However YoloV5 does not use it and instead requires a custom structure. This converter allows to convert any Coco dataset into a Yolo dataset, which can then be used for training.
All information required to convert a dataset can be found [here](coco2yolov5-converter/README.md).

The [official yolov5 repository](https://github.com/ultralytics/yolov5) contains the script to train a YoloV5 model.

