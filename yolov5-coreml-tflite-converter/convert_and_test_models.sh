#!/bin/bash

for s in "n" "n6" "s" "s6" "m" "m6" "l" "l6" "x" "x6"
do
    model=../../sbb-ml-yolov5-converter/data/yolov5_models/"yolov5"$s".pt"
    echo ""
    echo $s
    echo ""
    echo "model: "$model

    for c in "coreml" "tflite"
        do
            if [ $c == "tflite" ]; then
                python tflite/convert.py --model $model --quantize float32 float16 int8 --output-name "yolov5_"$s --out ../data/output/yolov5_models_converted/"yolov5_"$s
            fi
            if [ $c == "coreml" ]; then
                python coreml/convert.py --model $model --quantize float32 float16 int8 --output-name "yolov5_"$s --out ../data/output/yolov5_models_converted/"yolov5_"$s
            fi
    done
done



