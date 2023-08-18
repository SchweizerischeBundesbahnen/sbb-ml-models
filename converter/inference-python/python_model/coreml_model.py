import logging
import ast
import torch
from coremltools.models.model import MLModel
from helpers.coordinates import pt_yxyx2xyxy_yolo
from helpers.constants import BATCH_SIZE, IMAGE_NAME, IOU_NAME, CONF_NAME, CONFIDENCE_NAME, COORDINATES_NAME, \
    MASKS_NAME, NUMBER_NAME, PREDICTIONS_NAME, PROTOS_NAME, ULTRALYTICS, DETECTION, SEGMENTATION
from pytorch_utils.pytorch_nms import pt_xywh2yxyx_yolo
from python_model.inference_model_abs import InferenceModelAbs
from pytorch_utils.pytorch_nms import YoloNMS

class CoreMLModel(InferenceModelAbs):
    """ Class to load a CoreML model and run inference

        Attributes
        ----------
        model_path: str
            The path to the CoremL model
    """
    def __init__(self, model_path):
        logging.info("- Initializing CoreML model...")
        self.model = MLModel(model_path)
        self.__load_metadata()

    def predict(self, img, iou_threshold, conf_threshold):
        if self.do_nms:
            predictions = self.model.predict({IMAGE_NAME: img})

            preds = torch.from_numpy(predictions['var_1004'])

            # Postprocessing
            nms = YoloNMS(None, iou_thres=iou_threshold, conf_thres=conf_threshold, normalized=self.do_normalize, nmsed=self.do_nms, class_labels=self.labels)
            yxyx, classes, scores, masks = nms.nms_yolov5(preds, 'ultralytics')
            nb_detected = classes.shape[1]

            if self.model_type == SEGMENTATION:
                protos = torch.from_numpy(predictions[PROTOS_NAME])
                xyxy = pt_yxyx2xyxy_yolo(yxyx[0][:nb_detected])
                masks = process_mask(protos[0], masks[0], xyxy, self.img_size, upsample=True)  # HWC
            else:
                masks = None

            yxyx /= self.img_size[0]
            return yxyx, classes, scores, masks, torch.Tensor([nb_detected])

        else:
            try:
                predictions = self.model.predict(
                    {IMAGE_NAME: img, IOU_NAME: [iou_threshold], CONF_NAME: [conf_threshold]})
            except:
                predictions = self.model.predict(
                    {IMAGE_NAME: img, IOU_NAME: iou_threshold, CONF_NAME: conf_threshold})

            yxyx = torch.from_numpy(predictions[COORDINATES_NAME]).view(-1, 4)
            confidence = torch.from_numpy(predictions[CONFIDENCE_NAME]).view(-1, len(self.labels))

            yxyx = pt_xywh2yxyx_yolo(yxyx)  # coordinates are xywh
            classes = torch.argmax(confidence, dim=1)
            scores = torch.max(confidence, dim=1).values
            nb_detected = confidence.shape[0]

            if MASKS_NAME in predictions.keys():
                masks = torch.from_numpy(predictions[MASKS_NAME]).view(-1, self.img_size[0], self.img_size[1])
                nb_detected = predictions[NUMBER_NAME][0]
            else:
                masks = None

            return yxyx.unsqueeze(0), classes.unsqueeze(0), scores.unsqueeze(0), masks, torch.Tensor([nb_detected])

    def __load_metadata(self):
        spec = self.model.get_spec()
        pipeline = spec.pipeline
        metadata = self.model.user_defined_metadata

        self.model_type = metadata['task']
        self.model_orig = ULTRALYTICS if self.model_type == DETECTION else metadata['orig'] # Conversion with ultralytics change output shapes.
        self.labels = ast.literal_eval(metadata['names'])
        self.do_normalize = False
        self.do_nms = False if metadata['nms'] == 'True' else True
        self.quantization_type = metadata['quantization_type']

        self.batch_size = BATCH_SIZE
        self.pil_image = True
        self.channel_first = False

        input_details = spec.description.input
        output_details = spec.description.output
        self.input_dict = { input.name : f"image {input.type.imageType.height, input.type.imageType.width}" if input.name == IMAGE_NAME else f"double" for input in input_details}
        self.output_dict = { output.name : "array" for output in output_details}
        self.output_name = ', '.join([output.name for output in output_details])
        self.img_size = (int(input_details[0].type.imageType.height), int(input_details[0].type.imageType.width))
