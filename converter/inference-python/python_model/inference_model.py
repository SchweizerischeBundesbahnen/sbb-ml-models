import logging
from helpers.constants import BLUE, RED, END_COLOR, DEFAULT_INPUT_RESOLUTION, BATCH_SIZE, BOLD
from pathlib import Path
from python_model.coreml_model import CoreMLModel
from python_model.pytorch_model import PyTorchModel
from python_model.tflite_model import TFLiteModel
from python_model.onnx_model import ONNXModel

class InferenceModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.__init_model()

    def predict(self, img, iou_threshold, conf_threshold):
        return self.model.predict(img, iou_threshold, conf_threshold)

    def get_input_info(self):
        return self.model.do_normalize, self.model.img_size, self.model.batch_size, self.model.pil_image, self.model.channel_first

    def get_labels(self):
        return self.model.labels

    def __init_model(self):
        # Init model (TFLite, CoreML, PyTorch, ONNX)
        self.model_name = Path(self.model_path).name

        if not Path(self.model_path).exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{self.model_path}'")
            exit(0)

        logging.info(f'{BLUE}SETUP: finding the type of the model...{END_COLOR}')
        if self.model_name.endswith('.tflite'):
            logging.info(f'- The model is a {BOLD}TFLite{END_COLOR} model.')
            try:
                self.model = TFLiteModel(self.model_path)
                self.prefix = 'tflite'
            except ValueError as e:
                raise ValueError(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
        elif self.model_name.endswith('.mlmodel'):
            logging.info(f'- The model is a {BOLD}CoreML{END_COLOR} model.')
            try:
                self.model = CoreMLModel(self.model_path)
                self.prefix = 'coreml'
            except Exception as e:
                raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
        elif self.model_name.endswith('.pt'):
            logging.info(f'- The model is a {BOLD}PyTorch{END_COLOR} model.')
            try:
                self.model = PyTorchModel(self.model_path)
                self.prefix = 'pytorch'
            except Exception as e:
                raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
        elif self.model_name.endswith(".onnx"):
            logging.info(f'- The model is a {BOLD}ONNX{END_COLOR} model.')
            try:
                self.model = ONNXModel(self.model_path)
                self.prefix = 'onnx'
            except Exception as e:
                raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
        else:
            logging.info(
                f"{RED}Model format not supported:{END_COLOR} {self.model_name}. Supported format: .mlmodel, .onnx, .tflite, .pt.")
            exit(0)

        self.model.print_model_info()