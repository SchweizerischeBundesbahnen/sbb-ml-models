rm -rf yolov5 && git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5
cp yolov5.setup.py yolov5/setup.py
cp yolov5.yolo.py yolov5/models/yolo.py
cp yolov5.tf.py yolov5/models/tf.py

pip3 install -r requirements.txt
pip3 install -e converter_utils
pip3 install -e yolov5
pip3 install -e tflite
