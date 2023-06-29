import logging
from pathlib import Path

import cv2
import numpy as np

from constants import DEFAULT_INPUT_RESOLUTION, BLUE, END_COLOR, RED, GREEN, BOLD, PURPLE, DEFAULT_DETECTED_IMAGE_DIR
from detect import Detector


class InferenceTest:
    def __init__(self, pt_model_path, img_dir_path):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
        self.pt_model_path = Path(pt_model_path)
        self.img_dir_path = Path(img_dir_path)

        if not self.pt_model_path.exists():
            print(f"{RED}Pytorch model not found:{END_COLOR} '{self.pt_model_path}'")
            exit(1)

        if not self.img_dir_path.exists():
            print(f"{RED}Data directory not found:{END_COLOR} '{self.img_dir_path}'")
            exit(1)

        self.init_pytorch = False

    def __inference_pytorch(self, pt_input_resolution=DEFAULT_INPUT_RESOLUTION, save_img=False):
        detector = Detector(self.pt_model_path, pt_input_resolution=pt_input_resolution)
        self.pt_input_resolution = pt_input_resolution
        self.pt_detections, self.pt_inference_times, self.pt_image_names, self.pt_imgs_annotated = detector.detect(
            img_dir=self.img_dir_path,
            save_img=False,
            return_image=save_img,
            verbose=False)
        self.init_pytorch = True

    def run_test(self, model_path, save_img=False, out_path=DEFAULT_DETECTED_IMAGE_DIR):
        if not Path(model_path).exists():
            raise ValueError(f"{RED}Error:{END_COLOR} Converted model not found ({model_path})")

        logging.info(f"{PURPLE}{BOLD}Testing model: {model_path}{END_COLOR}")
        detector = Detector(model_path)
        detections, inference_times, image_names, imgs_annotated = detector.detect(img_dir=self.img_dir_path,
                                                                                   save_img=False,
                                                                                   return_image=save_img,
                                                                                   verbose=False)

        if not self.init_pytorch:
            logging.info(f"{BLUE}Computing PyTorch detection with input resolution: {detector.img_size[0]}{END_COLOR}")
            self.__inference_pytorch(detector.img_size[0], save_img)
        elif detector.img_size[0] != self.pt_input_resolution:
            logging.info(
                f"{BLUE}Recomputing PyTorch detection with input resolution: {detector.img_size[0]}{END_COLOR}")
            self.__inference_pytorch(detector.img_size[0], save_img)

        for i, (detection, pt_detection) in enumerate(zip(detections, self.pt_detections)):
            display_detections_pytorch(i, detection, image_names[i], pt_detection)

        if save_img:
            logging.info(f'{BLUE}Saving annotated images to {out_path}.{END_COLOR}')
            out_path = Path(out_path)
            if not out_path.exists():
                out_path.mkdir(exist_ok=True, parents=True)

            for image_name, conv_img, pt_img in zip(image_names, imgs_annotated, self.pt_imgs_annotated):
                model_name = Path(model_path).name
                new_image = np.concatenate([conv_img, pt_img], axis=1)
                out_path_img = str(out_path / f"{image_name.rsplit('.')[0]}_boxes_{model_name.rsplit('.')[0]}.png")
                cv2.imwrite(out_path_img, new_image)

        logging.info(f"{BLUE}Comparing with PyTorch model{END_COLOR}")
        self.__check_inference_time(inference_times, self.pt_inference_times)
        self.__check_detections(detections, self.pt_detections)

    @staticmethod
    def __check_detections(actual_detections, expected_detections):
        not_detected = 0
        detected = 0
        add_detected = 0
        for act_det, exp_det in zip(actual_detections, expected_detections):
            for exp_obj, exp_nb in exp_det.items():
                if exp_obj in act_det.keys():
                    act_nb = act_det.get(exp_obj)
                    not_detected += max(exp_nb - act_nb, 0)
                    add_detected += max(act_nb - exp_nb, 0)
                    detected += min(act_nb, exp_nb)
                else:
                    not_detected += exp_nb
        logging.info(f"- Number of object detected: {GREEN}{detected}{END_COLOR}")
        logging.info(f"- Number of object not detected: {RED}{not_detected}{END_COLOR}")
        logging.info(f"- Number of additional object detected: {RED}{add_detected}{END_COLOR}")

        perf = detected / (not_detected + add_detected + detected)
        threshold = 0.8
        if perf > threshold:
            print(
                f"{GREEN}Test success:{END_COLOR} the percentage of detections ({perf:.3}) is high enough (> {threshold}).")

        else:
            print(f"{RED}Test failure:{END_COLOR} the percentage of detections ({perf:.3}) is too low (< {threshold}).")
            exit(1)

    @staticmethod
    def __check_inference_time(actual_times, expected_times):
        act_average = 0
        exp_average = 0
        for act_time, exp_time in zip(actual_times, expected_times):
            act_average += act_time
            exp_average += exp_time
        act_average /= len(actual_times)
        exp_average /= len(expected_times)
        logging.info(f"- Average time converted model: {BLUE}{act_average:.3f}{END_COLOR}")
        logging.info(f"- Average time pytorch model: {BLUE}{exp_average:.3f}{END_COLOR}")


def display_detections_pytorch(i, detection, image_name, pt_detection):
    logging.info(f"{BLUE}Image {i + 1}:{END_COLOR} ({image_name})")
    logging.info(f"{f'{BOLD}Converted model{END_COLOR}':40}{f'{BOLD}PyTorch model{END_COLOR}'}")
    logging.info(
        f"{f'{sum([v for v in detection.values()])} detected objects':35}{f'{sum([v for v in pt_detection.values()])} detected objects'}")
    max_nb_predictions = len(detection) if len(detection) >= len(pt_detection) else len(pt_detection)
    det_strings = [f"{'':35}" for _ in range(max_nb_predictions)]

    for j, (k, v) in enumerate(detection.items()):
        temp_string = f"- {v} {k}{'s' if v > 1 else ''}"
        det_strings[j] = f"{temp_string:35}"

    for j, (k, v) in enumerate(pt_detection.items()):
        temp_string = f"- {v} {k}{'s' if v > 1 else ''}"
        det_strings[j] += f"{temp_string}"

    for s in det_strings:
        logging.info(s)
