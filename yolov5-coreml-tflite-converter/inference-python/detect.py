import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import PIL
import cv2
import numpy as np
import torch
from utils.dataloaders import LoadImages

from helpers.constants import DEFAULT_IOU_THRESHOLD, DEFAULT_CONF_THRESHOLD, DEFAULT_DETECTED_IMAGE_DIR, \
    DEFAULT_INPUT_RESOLUTION, RED, BLUE, END_COLOR, NORMALIZATION_FACTOR
from python_model.coreml_model import CoreMLModel
from python_model.pytorch_model import PyTorchModel
from python_model.tflite_model import TFLiteModel
from python_utils.plots import plot_boxes, plot_masks

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


class Detector:

    def __init__(self, model_path, pt_input_resolution=DEFAULT_INPUT_RESOLUTION):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.model_path = model_path
        self.pt_input_resolution = pt_input_resolution

        self.__init_model()

    def __init_model(self):
        # Init model (TFLite, CoreML, PyTorch)
        self.model_name = Path(self.model_path).name

        if not Path(self.model_path).exists():
            logging.info(f"{RED}Model not found:{END_COLOR} '{self.model_path}'")
            exit(0)

        logging.info('SETUP: finding the type of the model...')
        if self.model_name.endswith('.tflite'):
            logging.info('- The model is a TFLite model.')
            self.prefix = 'tflite'
            try:
                self.model = TFLiteModel(self.model_path)
            except ValueError as e:
                raise ValueError(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
            self.do_normalize, self.img_size, self.batch_size, self.pil_image, self.channel_first = self.model.get_input_info()
            self.labels = self.model.get_labels()
        elif self.model_name.endswith('.mlmodel'):
            logging.info('- The model is a CoreML model.')
            self.prefix = 'coreml'
            try:
                self.model = CoreMLModel(self.model_path)
            except Exception as e:
                raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
            self.do_normalize, self.img_size, self.batch_size, self.pil_image, self.channel_first = self.model.get_input_info()
            self.labels = self.model.get_labels()
        elif self.model_name.endswith('.pt'):
            logging.info('- The model is a PyTorch model.')
            self.prefix = 'pytorch'
            try:
                self.model = PyTorchModel(self.model_path, self.pt_input_resolution)
            except Exception as e:
                raise Exception(f"{RED}An error occured while initializing the model:{END_COLOR} {e}")
            self.do_normalize, self.img_size, self.batch_size, self.pil_image, self.channel_first = self.model.get_input_info()
            self.labels = self.model.get_labels()
        else:
            logging.info(
                f"{RED}Model format not supported:{END_COLOR} {self.model_name}. Supported format: .mlmodel, .onnx, .tflite, .pt.")
            exit(0)

    def detect_image(self, img, iou_threshold=DEFAULT_IOU_THRESHOLD, conf_threshold=DEFAULT_CONF_THRESHOLD):
        img = img.float()

        if self.do_normalize:
            # Normalize image
            img = img.float() / NORMALIZATION_FACTOR

        if not self.channel_first:
            img = img.permute(1, 2, 0)

        if self.pil_image:
            img = PIL.Image.fromarray(img.numpy().astype(np.uint8), 'RGB')
        else:
            img = img.unsqueeze(0)

        # Inference
        start_time = time.time()
        yxyx, classes, scores, masks, nb_detected = self.model.predict(img, iou_threshold, conf_threshold)
        inference_time = time.time() - start_time

        yxyx = yxyx if isinstance(yxyx, torch.Tensor) else torch.from_numpy(yxyx)
        classes = classes if isinstance(classes, torch.Tensor) else torch.from_numpy(classes)
        scores = scores if isinstance(scores, torch.Tensor) else torch.from_numpy(scores)
        if masks is not None:
            masks = masks if isinstance(masks, torch.Tensor) else torch.from_numpy(masks)
        return yxyx, classes, scores, masks, nb_detected, inference_time

    def detect(self, img_dir, max_img=-1, out_path=DEFAULT_DETECTED_IMAGE_DIR,
               iou_threshold=DEFAULT_IOU_THRESHOLD,
               conf_threshold=DEFAULT_CONF_THRESHOLD, save_img=True, return_image=False, verbose=True):

        img_path = Path(img_dir)
        out_path = Path(out_path)

        if not img_path.exists():
            logging.info(f"{RED}Directory not found:{END_COLOR} {img_dir}.")
            exit(1)

        dataset = LoadImages(img_dir, img_size=self.img_size, auto=False)

        if not out_path.exists() and save_img:
            out_path.mkdir(exist_ok=True, parents=True)

        detections = []
        inference_times = []
        image_names = []
        imgs_annotated = []
        try:
            if verbose:
                logging.info(f"{BLUE}DETECTION START{END_COLOR}")
            for i, (img_path, img, img_orig, _, _) in enumerate(dataset):
                if max_img != -1 and (i + 1) * self.batch_size > max_img:
                    break
                img_name = Path(img_path).name
                image_names.append(img_name)
                if verbose:
                    logging.info(
                        f"{BLUE}Image {i + 1}:{END_COLOR} ({img_name}: {img_orig.shape[0]}x{img_orig.shape[1]})")

                img = torch.from_numpy(img)

                yxyx, classes, scores, masks, nb_detected, inference_time = self.detect_image(img, iou_threshold=iou_threshold,
                                                                                       conf_threshold=conf_threshold)

                end_time = time.time()
                inference_times.append(inference_time)

                # Plot the bounding box
                if save_img or return_image:
                    if masks is None:
                        img_annotated = plot_boxes(self.img_size, [img_orig], yxyx, classes, scores, nb_detected, self.labels)
                    else:
                        img_annotated = plot_masks(self.img_size, [img_orig], yxyx, classes, scores, masks, nb_detected, self.labels)
                end_plot_time = time.time()

                # Save the results
                out_path_img = str(out_path / f"{self.prefix}_{img_name.rsplit('.')[0]}_boxes_{self.model_name.rsplit('.')[0]}.png")
                if save_img:
                    cv2.imwrite(out_path_img, img_annotated)
                if return_image:
                    imgs_annotated.append(img_annotated)
                counter = get_counter_detections(self.labels, classes, nb_detected)
                if verbose:
                    logging.info(f"\t- {sum([v for v in counter.values()])} detected objects")

                    for k, v in counter.items():
                        logging.info(f"\t\t{v} {k}{'s' if v > 1 else ''}")

                detections.append({k: v for k, v in counter.items()})

                if verbose:
                    logging.info(
                        f"\t- It took {inference_time:.3f} seconds to run the inference")
                    if save_img:
                        logging.info(f"\t- It took {end_plot_time - end_time:.3f} seconds to plot the results.")
                        logging.info(f"The output is saved in {out_path_img}.")

        except IndexError as e:
            raise IndexError(f"An error occured during the detection: {e}")
        return detections, inference_times, image_names, imgs_annotated


def get_counter_detections(labels, classes, nb_detected):
    # Get the number of detections and their label
    nb_det = int(nb_detected[0])
    detected_objects = [labels[int(x)] for x in classes[0][:nb_det]]
    counter = Counter(detected_objects)
    return counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        required=True,
                        help=f"The path to the converted model (tflite or coreml).")
    parser.add_argument('--img-dir', type=str, required=True,
                        help=f"The path to the images.")
    parser.add_argument('--max-img', type=int, default=-1,
                        help="The number of images to predict (maximum) among all the images in the directory. Default: -1 (no limit: all images in the directory will be processed).")
    parser.add_argument('--out', type=str, default=DEFAULT_DETECTED_IMAGE_DIR,
                        help=f"The path to the output directory (where to save the results). Default: '{DEFAULT_DETECTED_IMAGE_DIR}'.")
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f'IoU threshold. Default: {DEFAULT_IOU_THRESHOLD}')
    parser.add_argument('--conf-threshold', type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f'Confidence threshold. Default: {DEFAULT_CONF_THRESHOLD}')
    parser.add_argument('--no-save', action='store_true', help="If set, does not save the images.")

    opt = parser.parse_args()

    detector = Detector(opt.model)
    detector.detect(img_dir=opt.img_dir, max_img=opt.max_img, out_path=opt.out,
                    iou_threshold=opt.iou_threshold, conf_threshold=opt.conf_threshold, save_img=not opt.no_save)
