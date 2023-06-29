import logging
from pathlib import Path
from typing import List

import coremltools as ct
from coreml_converter.torchscript_exporter import TorchscriptExporter
from coreml_converter.torchscript_to_coreml_converter import TorchscriptToRawCoreMLConverter
from coreml_model.coreml_export_layer import CoreMLExportLayerGenerator
from coreml_model.model_spec_generator import ModelSpecGenerator
from coreml_model.nms_model_spec_generator import NMSModelSpecGenerator

from helpers.constants import FLOAT32_SUFFIX, FLOAT16_SUFFIX, INT8_SUFFIX, FLOAT32, FLOAT16, INT8, COREML_SUFFIX, \
    DETECTION, IOU_NAME, CONF_NAME, \
    CONFIDENCE_NAME, COORDINATES_NAME, MASKS_NAME, NUMBER_NAME, DEFAULT_MAX_NUMBER_DETECTION
from helpers.constants import OUTPUT_DIR, DEFAULT_COREML_NAME, DEFAULT_QUANTIZATION_TYPE, \
    DEFAULT_INPUT_RESOLUTION, BLUE, END_COLOR, GREEN, RED
from pytorch_utils.pytorch_loader import PyTorchModelLoader


class PytorchToCoreMLConverter:
    """ Class to convert a Pytorch model to a Keras model, saved as saved_model or TFLite

    Attributes
    ----------
    model_input_path: String
        The path to the Pytorch model

    model_output_directory: String
        The path to the directory in which to save the model

    model_output_name: String
        The name of the converted model (without extension)

    quantization_types: List[str]
        The quantization types to use for the conversion

    input_resolution: int
        The input resolution of the image

    max_det: int
        The maximum number of detections that can be made by the CoreML model
    """

    def __init__(self, model_input_path: str,
                 model_output_directory: str = OUTPUT_DIR,
                 model_output_name: str = DEFAULT_COREML_NAME,
                 quantization_types: List[str] = None,
                 input_resolution: int = DEFAULT_INPUT_RESOLUTION,
                 max_det: int = DEFAULT_MAX_NUMBER_DETECTION):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_input_path = Path(model_input_path)
        self.model_output_directory_path = Path(model_output_directory)
        self.model_output_name = model_output_name

        if quantization_types is None:
            quantization_types = [DEFAULT_QUANTIZATION_TYPE]
        self.quantization_types = quantization_types
        self.input_resolution = input_resolution
        self.max_det = max_det

        self.__check_input_create_output()
        self.__init_pytorch_model()

    def convert(self):
        """
        Converts a PyTorch model to a CoreML model and saves it
        """
        # Get CoreML spec
        model_spec = self.__get_coreml_spec()

        # Export the model
        file_name = f'{self.model_output_name}_{self.input_resolution}'
        if self.model.model_type == DETECTION:
            file_name += "_detection"
        else:
            file_name += "_segmentation"

        model = ct.models.MLModel(model_spec.spec)
        for quantization_type in self.quantization_types:
            try:
                logging.info(f'{BLUE}Exporting model to CoreML {quantization_type}{END_COLOR}')
                model_temp = model
                if quantization_type == FLOAT32:
                    suffix = FLOAT32_SUFFIX
                elif quantization_type == FLOAT16:
                    suffix = FLOAT16_SUFFIX
                    model_temp = ct.models.neural_network.quantization_utils.quantize_weights(
                        model, nbits=16)
                elif quantization_type == INT8:
                    suffix = INT8_SUFFIX
                    model_temp = ct.models.neural_network.quantization_utils.quantize_weights(
                        model, nbits=8)
                else:
                    raise ValueError(f"The quantization type '{quantization_type}' is not supported.")
                converted_model_path = self.model_output_directory_path / Path(file_name + suffix + COREML_SUFFIX)

                model_temp.save(converted_model_path)
                logging.info(f'{GREEN}CoreML export success:{END_COLOR} saved as {converted_model_path}')
            except Exception as e:
                raise Exception(f'{RED}CoreML export failure: {e}{END_COLOR}')

    def __check_input_create_output(self):
        # Check input and create output
        if not self.model_input_path.exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{str(self.model_input_path)}'")
            exit(0)
        self.model_output_directory_path.mkdir(parents=True, exist_ok=True)

    def __init_pytorch_model(self):
        # Load the model
        logging.info(f'{BLUE}Loading PyTorch model...{END_COLOR}')
        self.model = PyTorchModelLoader(self.model_input_path, self.input_resolution).load()

    def __get_coreml_spec(self):
        if self.model.model_type == DETECTION:
            # Put the model in training, we will add the export layer in CoreML
            self.model.torch_model.model[-1].training = True

            # Produces torchscript
            # Input: image
            # Output: predictions (1, nA, nC, nC, nO), ...
            # nA = number of anchors, nC = input_resolution / strides[i], nO = number of outputs (#classes + 5)
            self.model.torchscript_model = TorchscriptExporter(self.model).export()

            # Convert torchscript to raw coreml model
            raw_coreml_model = TorchscriptToRawCoreMLConverter(self.model).convert()
            builder = ct.models.neural_network.NeuralNetworkBuilder(spec=raw_coreml_model)

            # Add export logic to coreml model
            # Input: predictions (1, nA, nC, nC, nO), ...
            # Output: confidence, coordinates (nbPred, nbClass), (nbPred, 4)
            CoreMLExportLayerGenerator(self.model).add_to(builder)

            # Create nms logic
            # Input: confidence, coordinates, IoU and conf thresholds
            # Output: confidence, coordinates
            nms_spec = NMSModelSpecGenerator(self.model).generate(builder.spec)

            # Combine model with export logic and nms logic
            # Input: image, IoU and conf threshold
            # Output: confidence, coordinates
            model_spec = ModelSpecGenerator(self.model).generate(builder.spec, nms_spec)

            return model_spec
        else:
            self.model.torch_model.model[-1].export = True
            # Produces torchscript
            # Input: image
            # Output: predictions (1, #predictions, nO), (1, nM, H, W)
            # #predictions = number of unfiltered predictions, nO = number of outputs (#classes + 5 + nM)
            # nM = number of masks, H = input img height / 4, W = input img height / 4
            self.model.torchscript_model = TorchscriptExporter(self.model).export()

            # Convert torchscript to raw coreml model
            raw_coreml_model = TorchscriptToRawCoreMLConverter(self.model).convert()

            # Create new model
            nb_predictions = sum(
                [len(self.model.anchors[i]) * (x ** 2) for i, x in enumerate(self.model.feature_map_dimensions)])
            input_features = [('all_predictions', ct.models.datatypes.Array(1, nb_predictions,
                                                                            self.model.number_of_classes + 5 + self.model.number_of_masks)),
                              ('segmentation_protos', ct.models.datatypes.Array(1, self.model.number_of_masks,
                                                                                self.model.input_resolution // 4,
                                                                                self.model.input_resolution // 4)),
                              (IOU_NAME, ct.models.datatypes.Array(1, )),
                              (CONF_NAME, ct.models.datatypes.Array(1, )), ]
            output_features = [(CONFIDENCE_NAME, None), (COORDINATES_NAME, None), (MASKS_NAME, None),
                               (NUMBER_NAME, None)]
            builder = ct.models.neural_network.NeuralNetworkBuilder(input_features=input_features,
                                                                    output_features=output_features,
                                                                    disable_rank5_shape_mapping=True)
            # Add export logic
            # Input: predictions, protos, iou, conf (1, #predictions, nO), (1, nM, H, W), (1,), (1,)
            # Output: confidence, coordinates, masks, nb detections (#detections, nbClass), (#detections, 4), (#detections, nM)
            CoreMLExportLayerGenerator(self.model).add_to(builder)

            # Combine model with export logic
            # Input: image, IoU and conf threshold
            # Output: confidence, coordinates, masks, nb detections
            model_spec = ModelSpecGenerator(self.model).generate(raw_coreml_model, builder.spec)

            return model_spec
