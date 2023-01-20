echo "Cloning \033[0;36mYoloV5\033[0m repo..."
rm -rf yolov5 && git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5
echo "Moving files to \033[0;36mYoloV5\033[0m repo..."
cp yolov5.setup.py yolov5/setup.py
cp yolov5.tf.py yolov5/models/tf.py
cp yolov5.common.py yolov5/models/common.py
echo "\033[0;32mYoloV5 repo is ready to be used!\033[0m"

echo "Installing all required packages..."
pip3 install -r requirements.txt
pip3 install -e converter_utils
pip3 install -e tflite
pip3 install -e yolov5
echo "\033[0;32mSuccessfully installed all packages!\033[0m"
