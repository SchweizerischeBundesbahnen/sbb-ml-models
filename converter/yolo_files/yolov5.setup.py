from setuptools import setup

setup(
    name="yolov5",
    url="https://github.com/ultralytics/yolov5",
    maintainer="ultralytics",
    maintainer_email="glenn.jocher@ultralytics.com",
    packages=["models", "utils"],
    install_requires=["opencv-python", "matplotlib", "torchvision", "PyYAML", "requests", "pandas", "seaborn"],
)
