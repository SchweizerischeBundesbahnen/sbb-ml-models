import argparse
import logging
from pathlib import Path

from helpers.constants import RED, END_COLOR, DEFAULT_MODEL_OUTPUT_DIR
from test_model import ModelTest


def test_tflite(reference_model, model_dir, img_dir, verbose):
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("\n----------------------------------------------------------------------")
    logging.info("TFLITE Test")

    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        logging.info(f"{RED}Directory not found{END_COLOR}: '{model_dir}'")
        exit(1)

    models = [str(m) for m in model_dir_path.glob('*.tflite')]

    for model in models:
        ModelTest(model, reference_model, img_dir).run(verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-model', required=True, type=str, help="The path to the original pytorch model.")
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_OUTPUT_DIR, type=str,
                        help="The path to the directory that contains the converted models.")
    parser.add_argument('--img-dir', required=True, type=str,
                        help="The path to the directory containing the images")
    parser.add_argument('--verbose', action='store_true',
                        help="If set, will print predictions and results for all classes.")

    opt = parser.parse_args()

    test_tflite(opt.reference_model, opt.model_dir, opt.img_dir, opt.verbose)
