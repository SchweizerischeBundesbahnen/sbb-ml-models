import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from constants import RED, END_COLOR, BLUE, DEFAULT_DETECTED_IMAGE_DIR, PURPLE, BOLD, GREEN
from coordinates import pt_yxyx2xyxy
from detect import Detector, get_counter_detections
from python_utils.plots import plot_boxes
from utils.datasets import LoadImages
from utils.metrics import ap_per_class
from val import process_batch


class ComparisonTest:
    def __init__(self, reference_model_path, data_path):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.data_path = data_path
        self.reference_model_path = reference_model_path

        if not Path(self.data_path).exists():
            print(f"{RED}Data directory not found:{END_COLOR} '{self.data_path}'")
            exit(1)

        if not Path(self.reference_model_path).exists():
            print(f"{RED}Model not found:{END_COLOR} '{self.reference_model_path}'")
            exit(1)

    def __check_classes(self, class_labels, reference_class_labels):
        if len(class_labels) != len(reference_class_labels):
            raise ValueError(
                f"{RED}Model error:{END_COLOR} the reference model and compared model do not consider the same classes.")
        else:
            for c in class_labels:
                if c not in reference_class_labels:
                    raise ValueError(
                        f"{RED}Model error:{END_COLOR} the reference model and compared model do not consider the same classes.")

    def __run_inference(self, i, img_path, img, reference_img):
        img = torch.from_numpy(img)
        reference_img = torch.from_numpy(reference_img)
        yxyx, classes, scores, nb_detected, inference_time = self.detector.detect_image(img)
        xyxy = pt_yxyx2xyxy(yxyx)

        reference_yxyx, reference_classes, reference_scores, reference_nb_detected, reference_inference_time = self.detector.detect_image(
            reference_img)
        reference_xyxy = pt_yxyx2xyxy(yxyx)

        if self.verbose:
            image_name = Path(img_path).name
            counter = get_counter_detections(self.class_labels, classes, nb_detected)
            reference_counter = get_counter_detections(self.class_labels, reference_classes, reference_nb_detected)
            self.display_detections(i, counter, reference_counter, image_name)

        # Consider each image
        for i in range(self.detector.batch_size):
            self.seen += 1

            # Compared model result (Nx6)
            predn = torch.cat([xyxy[i], scores[i].unsqueeze(1), classes[i].unsqueeze(1)], dim=1)
            # Reference model result (Nx6)
            reference_predn = torch.cat([reference_classes[i].unsqueeze(1), reference_xyxy[i]], dim=1)

            self.inference_time[0].append(inference_time)
            self.inference_time[1].append(reference_inference_time)

            correct = process_batch(predn, reference_predn, self.iouv)
            self.stats.append((correct, predn[:, 4], predn[:, 5], reference_predn[:, 0]))

    def run_test(self, compared_model_path, verbose=False):
        if not Path(compared_model_path).exists():
            print(f"{RED}Converted model not found:{END_COLOR} '{compared_model_path}'")
            exit(1)
        self.verbose = verbose
        self.compared_model_path = compared_model_path

        logging.info(f"{PURPLE}{BOLD}Testing model: {compared_model_path}{END_COLOR}")
        self.reference_detector = Detector(self.reference_model_path)
        self.detector = Detector(compared_model_path)

        # The reference and compared model should consider the same classes
        self.__check_classes(self.reference_detector.labels, self.detector.labels)
        self.class_labels = self.detector.labels

        dataset = LoadImages(self.data_path, img_size=self.detector.img_size, auto=False)

        self.stats = []
        self.inference_time = [[], []]
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.seen = 0

        if self.reference_detector.img_size != self.detector.img_size:
            reference_dataset = LoadImages(self.data_path, img_size=self.reference_detector.img_size, auto=False)
            for i, ((img_path, img, img_orig, _), (_, reference_img, _, _)) in enumerate(
                    zip(dataset, reference_dataset)):
                self.__run_inference(i, img_path, img, reference_img)
        else:
            dataset = LoadImages(self.data_path, img_size=self.detector.img_size, auto=False)
            for i, (img_path, img, img_orig, _) in enumerate(dataset):
                self.__run_inference(i, img_path, img, img)

        ct = np.mean(self.inference_time[0])
        rt = np.mean(self.inference_time[1])

        logging.info(
            f"{BLUE}{BOLD}Results for the compared model ({Path(self.compared_model_path).name}) w.r.t the reference model ({Path(self.reference_model_path).name}){END_COLOR}")
        logging.info(f"Average inference time (converted model): {BOLD}{ct:.3}s{END_COLOR}")
        logging.info(f"Average inference time (reference model): {rt:.3}s")

        map_threshold = 0.6
        success = True
        stat = [np.concatenate(x, 0) for x in zip(*self.stats)]
        if len(stat) and stat[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stat, plot=False, names=self.class_labels)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stat[3].astype(np.int64), minlength=len(self.class_labels))  # number of targets per class

            logging.info(f"{'Class':30}{'Images':11}{'Labels':11}{'P':11}{'R':11}{'mAP@.5':11}{'mAP@.5:.95':11}")
            logging.info(
                f"{BOLD}{'all':30}{self.seen:<11}{nt.sum():<11}{mp:<11.3}{mr:<11.3}{map50:<11.3}{map:<11.3}{END_COLOR}")

            if map < map_threshold:
                success = False

            if verbose:
                # Print results per class
                for i, c in enumerate(ap_class):
                    logging.info(
                        f"{self.class_labels[c]:30}{self.seen:<11}{nt[c]:<11}{p[i]:<11.3}{r[i]:<11.3}{ap50[i]:<11.3}{ap[i]:<11.3}")
        else:
            success = False

        if success:
            logging.info(
                f"{GREEN}Test success:{END_COLOR} the model passes the test. The mAP score w.r.t. the reference model is above {map_threshold}.")
        else:
            logging.info(f"{RED}Test failure:{END_COLOR} the converted model is not good enough.")

    @staticmethod
    def display_detections(i, detection, reference_detection, image_name):
        logging.info(f"{BLUE}Image {i + 1}:{END_COLOR} ({image_name})")
        logging.info(f"{f'{BOLD}Compared model{END_COLOR}':40}{f'{BOLD}Reference model{END_COLOR}'}")
        logging.info(
            f"{f'{sum([v for v in detection.values()])} detected objects':35}{f'{sum([v for v in reference_detection.values()])} detected objects'}")
        max_nb_predictions = len(detection) if len(detection) >= len(reference_detection) else len(reference_detection)
        det_strings = [f"{'':35}" for _ in range(max_nb_predictions)]

        for j, (k, v) in enumerate(detection.items()):
            temp_string = f"- {v} {k}{'s' if v > 1 else ''}"
            det_strings[j] = f"{temp_string:35}"

        for j, (k, v) in enumerate(reference_detection.items()):
            temp_string = f"- {v} {k}{'s' if v > 1 else ''}"
            det_strings[j] += f"{temp_string}"

        for s in det_strings:
            logging.info(s)
