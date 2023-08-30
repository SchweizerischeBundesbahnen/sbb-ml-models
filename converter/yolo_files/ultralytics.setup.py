from setuptools import setup

setup(
    name="ultralytics",
    url="https://github.com/ultralytics/ultralytics",
    maintainer="ultralytics",
    maintainer_email="glenn.jocher@ultralytics.com",
    packages=["ultralytics.nn", "ultralytics.engine"],
    install_requires=["omegaconf"],
)
