echo "Installing all required packages..."
conda install tensorflow=2.11.0

pip3 install -r requirements_m1.txt
pip3 install -e converter_utils
pip3 install -e tflite
pip3 install -e yolov5

rm -rf tflite-support && git clone https://github.com/tensorflow/tflite-support.git

cd tflite-support
bazel build -c opt tensorflow_lite_support/tools/pip_package:build_pip_package
./bazel-bin/tensorflow_lite_support/tools/pip_package/build_pip_package --dst wheels --nightly_flag
pip3 install wheels/*
cd ..

echo "\033[0;32mSuccessfully installed all packages!\033[0m"



