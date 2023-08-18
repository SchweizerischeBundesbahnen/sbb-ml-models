import coremltools as ct
from coreml_converter.torchscript_exporter import TorchscriptExporter
from coreml_converter.torchscript_to_coreml_converter import TorchscriptToRawCoreMLConverter
from coreml_model.coreml_export_layer import CoreMLSegmentationExportLayerGenerator
from coreml_model.model_spec_generator import ModelSpecGenerator
from coremltools.proto import Model_pb2
from helpers.constants import BATCH_SIZE, NB_CHANNEL, IOU_NAME, CONF_NAME, CONFIDENCE_NAME, COORDINATES_NAME, \
    MASKS_NAME, NUMBER_NAME


class CoreMLSegmentationModelSpec:
    """ Class that creates the specifications for Yolo Segmentation model

        Attributes
        ----------
        model: ModelWrapper
            The model to be converted to CoreML
        """

    def __init__(self, pt_model):
        self.pt_model = pt_model
        self.pt_model.class_labels = self.pt_model.torch_model.names
        self.pt_model.number_of_classes = len(self.pt_model.class_labels)
        self.pt_model.input_shape = (BATCH_SIZE, NB_CHANNEL, self.pt_model.model_parameters.input_resolution,
                                     self.pt_model.model_parameters.input_resolution)

    def generate_specs(self) -> Model_pb2:
        """ Generates new CoreML model from PyTorch model + NMS

        Returns
        ----------
        nn_spec: Model_pb2
            The specification of the CoreML model
        """
        # Produces torchscript
        # Input: image
        # Output: predictions (1, #predictions, nO), (1, nM, H, W)
        # #predictions = number of unfiltered predictions, nO = number of outputs (#classes + 5 + nM)
        # nM = number of masks, H = input img height / 4, W = input img height / 4
        self.pt_model.torch_model.model[-1].export = True
        self.pt_model.torch_model.model[-1].format = 'coreml'
        self.pt_model.torchscript_model = TorchscriptExporter(self.pt_model).export()

        # Convert torchscript to raw coreml model
        raw_coreml_model = TorchscriptToRawCoreMLConverter(self.pt_model).convert()

        # Create new model
        input_features = [('all_predictions', ct.models.datatypes.Array(
            *raw_coreml_model.description.output[0].type.multiArrayType.shape)),
                          ('segmentation_protos',
                           ct.models.datatypes.Array(
                               *raw_coreml_model.description.output[1].type.multiArrayType.shape)),
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
        CoreMLSegmentationExportLayerGenerator(self.pt_model).add_to(builder)

        # Combine model with export logic
        # Input: image, IoU and conf threshold
        # Output: confidence, coordinates, masks, nb detections
        model_spec = ModelSpecGenerator(self.pt_model).generate(raw_coreml_model, builder.spec)
        return model_spec
