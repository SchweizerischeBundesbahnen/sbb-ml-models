echo "Cloning \033[0;36mYoloV5\033[0m repo..."
rm -rf yolov5 && git clone -b v7.0 --depth 1 https://github.com/ultralytics/yolov5
echo "Moving files to \033[0;36mYoloV5\033[0m repo..."
cp yolov5.setup.py yolov5/setup.py
cp yolov5.tf.py yolov5/models/tf.py
cp yolov5.common.py yolov5/models/common.py
echo "\033[0;32mYoloV5 repo is ready to be used!\033[0m"


echo "Creating conda environment..."
conda create -n converter_yolo python=3.8
echo "Now run the command: \033[0;33mconda activate converter_yolo\033[0m to activate the environment, then use \033[0;33m./install_dependencies_m1.sh\033[0m to install all the dependencies."


