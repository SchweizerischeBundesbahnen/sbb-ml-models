import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from utils.datasets import LoadImagesAndLabels
from utils.general import check_dataset
from utils.metrics import ap_per_class
from val import process_batch

from constants import RED, END_COLOR, BOLD, BLUE, PURPLE, GREEN, DEFAULT_DETECTED_IMAGE_DIR
from coordinates import pt_yxyx2xyxy, pt_xywh2xyxy_yolo
from detect import Detector, get_counter_detections
from python_utils.plots import plot_boxes
from test_utils.test_inference import display_detections_pytorch


class AccuracyTest:
    def __init__(self, pt_model_path, dataset):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.pt_model_path = Path(pt_model_path)
        self.dataset = check_dataset(dataset)

        if 'test' in self.dataset.keys():
            self.img_dir_path = Path(self.dataset['test'])
        else:
            raise ValueError(f"{RED}Dataset error:{END_COLOR} The dataset does not contain a test set.")

        if not self.pt_model_path.exists():
            print(f"{RED}Pytorch model not found:{END_COLOR} '{self.pt_model_path}'")
            exit(1)

        if not self.img_dir_path.exists():
            raise ValueError(f"{RED}Dataset error:{END_COLOR} The test image dir does not exist ({self.img_dir_path})")

        self.init_pytorch = False

    def __check_classes(self, class_labels):
        pt_class_labels = Detector(self.pt_model_path).labels
        if len(class_labels) != len(pt_class_labels):
            raise ValueError(
                f"{RED}Model error:{END_COLOR} the pytorch model and converted model do not consider the same classes.")
        else:
            for c in class_labels:
                if c not in pt_class_labels:
                    raise ValueError(
                        f"{RED}Model error:{END_COLOR} the pytorch model and converted model do not consider the same classes.")

        if len(class_labels) != int(self.dataset['nc']):
            raise ValueError(
                f"{RED}Dataset error:{END_COLOR} the dataset does not contain the same classes as the models.")
        else:
            for c in class_labels:
                if c not in self.dataset['names']:
                    raise ValueError(
                        f"{RED}Dataset error:{END_COLOR} the dataset does not contain the same classes as the models.")

    def __inference_pytorch(self, pt_input_resolution):
        self.pt_input_resolution = pt_input_resolution

        detector = Detector(self.pt_model_path, pt_input_resolution=pt_input_resolution)
        img_size = detector.img_size
        batch_size = detector.batch_size

        dataset = LoadImagesAndLabels(self.img_dir_path, img_size[0], batch_size)

        self.pytorch_detections = []
        for i, (img, targets, paths, _) in enumerate(dataset):
            pt_yxyx, pt_classes, pt_scores, pt_nb_detected, pt_inference_time = detector.detect_image(img)

            self.pytorch_detections.append((pt_yxyx, pt_classes, pt_scores, pt_nb_detected, pt_inference_time))

        self.init_pytorch = True

    def run_test(self, model_path, verbose=False, save_imgs=False, out_path=DEFAULT_DETECTED_IMAGE_DIR):
        if not Path(model_path).exists():
            print(f"{RED}Converted model not found:{END_COLOR} '{model_path}'")
            exit(1)

        logging.info(f"{PURPLE}{BOLD}Testing model: {model_path}{END_COLOR}")
        detector = Detector(model_path)
        class_labels = detector.labels
        img_size = detector.img_size
        batch_size = detector.batch_size
        model_name = Path(model_path).name

        # Pytorch model, converted model and dataset should consider the same classes
        self.__check_classes(class_labels)

        if not self.init_pytorch:
            logging.info(f"{BLUE}Computing PyTorch detection with input resolution: {detector.img_size[0]}{END_COLOR}")
            self.__inference_pytorch(detector.img_size[0])
        elif detector.img_size[0] != self.pt_input_resolution:
            logging.info(
                f"{BLUE}Recomputing PyTorch detection with input resolution: {detector.img_size[0]}{END_COLOR}")
            self.__inference_pytorch(detector.img_size[0])

        dataset = LoadImagesAndLabels(self.img_dir_path, img_size[0], batch_size)
        # Comparison between converted model and ground truth
        # Comparison between converted model and pytorch model
        stats = [[], [], []]
        inference_times = [[], []]

        iouv = torch.linspace(0.5, 0.95, 10)
        seen = 0
        # Consider a batch of images
        for i, (img, targets, paths, _) in enumerate(dataset):
            yxyx, classes, scores, nb_detected, inference_time = detector.detect_image(img)
            xyxy = pt_yxyx2xyxy(yxyx)

            pt_yxyx, pt_classes, pt_scores, pt_nb_detected, pt_inference_time = self.pytorch_detections[i]
            pt_xyxy = pt_yxyx2xyxy(pt_yxyx)

            if verbose:
                image_name = Path(paths).name
                counter = get_counter_detections(class_labels, classes, nb_detected)
                pt_counter = get_counter_detections(class_labels, pt_classes, pt_nb_detected)
                display_detections_pytorch(i, counter, image_name, pt_counter)

            if save_imgs:
                logging.info(f'{BLUE}Saving annotated images to {out_path}.{END_COLOR}')
                image_name = Path(paths).name
                out_path = Path(out_path)
                if not out_path.exists():
                    out_path.mkdir(exist_ok=True, parents=True)

                img_orig = cv2.imread(paths)

                img_annotated = img_orig.copy()
                plot_boxes(img_size, [img_annotated], yxyx, classes, scores, nb_detected, class_labels)

                img_annotated_pytorch = img_orig.copy()
                plot_boxes(img_size, [img_annotated_pytorch], pt_yxyx, pt_classes, pt_scores, pt_nb_detected,
                           class_labels)

                imgs_annotated = np.concatenate([img_annotated, img_annotated_pytorch], axis=1)

                out_path_img = str(out_path / f"{image_name.rsplit('.')[0]}_boxes_{model_name.rsplit('.')[0]}.png")

                cv2.imwrite(out_path_img, imgs_annotated)

            # Consider each image
            for i in range(batch_size):
                seen += 1
                # Get the target labels and bbox for that image
                current_target = targets[targets[:, 0] == i, 1:]
                target_labels = current_target[:, 0]
                target_xywh = current_target[:, 1:]
                target_xyxy = pt_xywh2xyxy_yolo(target_xywh)

                # Converted model result (Nx6)
                predn = torch.cat([xyxy[i], scores[i].unsqueeze(1), classes[i].unsqueeze(1)], dim=1)
                # Pytorch model result (Nx6)
                pt_predn = torch.cat([pt_xyxy[i], pt_scores[i].unsqueeze(1), pt_classes[i].unsqueeze(1)], dim=1)
                # Ground truth (Nx5)
                labelsn = torch.cat([target_labels.unsqueeze(1), target_xyxy], dim=1)
                # Pytorch model result (Nx5)
                pt_labelsn = torch.cat([pt_classes[i].unsqueeze(1), pt_xyxy[i]], dim=1)

                inference_times[0].append(inference_time)
                inference_times[1].append(pt_inference_time)

                for i, (act, exp) in enumerate([(pt_predn, labelsn), (predn, labelsn), (predn, pt_labelsn)]):
                    correct = process_batch(act, exp, iouv)
                    stats[i].append((correct, act[:, 4], act[:, 5], exp[:, 0]))

        ct = np.mean(inference_times[0])
        pt = np.mean(inference_times[1])

        logging.info(f"{BLUE}{BOLD}Results ({model_path}){END_COLOR}")
        logging.info(f"Average inference time (converted model): {BOLD}{ct:.3}s{END_COLOR}")
        logging.info(f"Average inference time (pytorch model): {pt:.3}s")
        s = ['Results for the pytorch model w.r.t. the ground truth',
             'Results for the converted model w.r.t. the ground truth',
             'Results for the converted model w.r.t. the pytorch model']
        success = [True, True, True]
        # First is overall performance of converted model
        # Second is performance of converted vs original
        map_thresholds = [0.3, 0.3, 0.7]

        for i, stat in enumerate(stats):
            logging.info(f"{BLUE}{s[i]}{END_COLOR}")
            stat = [np.concatenate(x, 0) for x in zip(*stat)]
            if len(stat) and stat[0].any():
                p, r, ap, f1, ap_class = ap_per_class(*stat, plot=False, names=class_labels)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stat[3].astype(np.int64), minlength=len(class_labels))  # number of targets per class

                logging.info(f"{'Class':30}{'Images':11}{'Labels':11}{'P':11}{'R':11}{'mAP@.5':11}{'mAP@.5:.95':11}")
                logging.info(
                    f"{BOLD}{'all':30}{seen:<11}{nt.sum():<11}{mp:<11.3}{mr:<11.3}{map50:<11.3}{map:<11.3}{END_COLOR}")

                if map < map_thresholds[i]:
                    success[i] = False

                if verbose:
                    # Print results per class
                    for i, c in enumerate(ap_class):
                        logging.info(
                            f"{class_labels[c]:30}{seen:<11}{nt[c]:<11}{p[i]:<11.3}{r[i]:<11.3}{ap50[i]:<11.3}{ap[i]:<11.3}")
            else:
                success = [False, False, False]
        if all(success):
            logging.info(f"{GREEN}Test success:{END_COLOR} the converted model passes the test. The mAP score w.r.t. the ground truth is above {map_thresholds[1]} and w.r.t. the pytorch model above {map_thresholds[2]}.")
        else:
            msg = f"{RED}Test failure:{END_COLOR} the converted model is not good enough. "
            if success[2]:
                msg += f"The conversion went well, meaning that the original pytorch model may not perform well. The mAP score w.r.t. the ground truth is bellow {map_thresholds[1]} but the mAP score w.r.t. to the pytorch model is higher than {map_thresholds[2]}, suggesting that the conversion went well."
            else:
                msg += f"The problem seem to come from the conversion which did not go well. The mAP score w.r.t. the pytorch model is below {map_thresholds[2]}."
            logging.info(msg)
            exit(1)
