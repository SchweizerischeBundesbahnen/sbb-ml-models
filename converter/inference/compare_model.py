import argparse

from utils_test.comparison_test import ComparisonTest


class ModelTest:
    def __init__(self, model_path, reference_model_path, data_path):
        self.model_path = model_path
        self.reference_model_path = reference_model_path
        self.data_path = data_path

    def run(self, verbose=True):
        ComparisonTest(self.reference_model_path, self.data_path).run_test(self.model_path, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, dest='model_path', help=f"The path to the compared model.")
    parser.add_argument('--reference-model', type=str,
                        required=True, dest='reference_model_paths',
                        help=f"The path to the reference model.")
    parser.add_argument('--img-dir', type=str, required=True,
                        help="The path to the directory containing the images")
    parser.add_argument('--verbose', action='store_true',
                        help="If set, will print predictions and results for all classes.")
    opt = parser.parse_args()

    ModelTest(opt.model_path, opt.reference_model_paths, opt.img_dir).run(opt.verbose)
