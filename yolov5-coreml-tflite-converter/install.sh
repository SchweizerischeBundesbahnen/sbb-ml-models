rm -rf yolov5 && git clone -b v6.0 --depth 1 https://github.com/ultralytics/yolov5
cp yolov5.setup.py yolov5/setup.py
cp yolov5.yolo.py yolov5/models/yolo.py
cp yolov5.tf.py yolov5/models/tf.py
pipenv install -r requirements.txt --skip-lock
pipenv install -e ./yolov5 --skip-lock
pipenv install -e ./converter_utils --skip-lock
pipenv install -e ./tflite
