conda install tensorflow=2.11.0

pip3 install torch==1.13.0
pip3 install torchvision==0.14.0
pip3 install coremltools==6.1
pip3 install protobuf==3.19.6
pip3 install psutil, tqdm, IPython
pip3 install -e converter_utils
pip3 install -e yolov5
pip3 install numpy==1.23.5

git clone https://github.com/tensorflow/tflite-support.git

cd tflite-support
bazel build -c opt tensorflow_lite_support/tools/pip_package:build_pip_package
./bazel-bin/tensorflow_lite_support/tools/pip_package/build_pip_package --dst wheels --nightly_flag
pip3 install wheels/*
cd ..




