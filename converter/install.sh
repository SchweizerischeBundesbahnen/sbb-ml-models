echo "Cloning \033[0;36mYoloV5\033[0m repo..."
rm -rf yolov5 && git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5
echo "Moving files to \033[0;36mYoloV5\033[0m repo..."
cp yolo_files/yolov5.setup.py yolov5/setup.py
cp yolo_files/yolov5.tf.py yolov5/models/tf.py
cp yolo_files/yolov5.common.py yolov5/models/common.py
echo "\033[0;32mYoloV5 repo is ready to be used!\033[0m"

echo "Installing all required packages..."
pip3 install -r requirements.txt
pip3 install -e converter_utils
pip3 install -e tflite
pip3 install -e yolov5
pip3 install -e ultralytics
echo "\033[0;32mSuccessfully installed all packages!\033[0m"

echo "Cloning \033[0;36mUltralytics\033[0m repo..."
rm -rf ultralytics && git clone https://github.com/ultralytics/ultralytics
cp yolo_files/ultralytics.setup.py ultralytics/setup.py
cd ultralytics
git checkout c20d2654e95d4d8f1a42e106118f21ddb2762115
echo "\033[0;32mUltralytics repo is ready to be used!\033[0m"
