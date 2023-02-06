import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

from helpers.constants import DATA_DIR, get_zipfile_path, get_dataset_url
from utils.dataloaders import LoadImages


class RepresentativeDatasetGenerator:
    """ Dataset generator for the representative dataset used during quantization

    Attributes
    ----------
    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. type, use nms, ...)

    conversion_parameters: ConversionParameters
        The parameters for the conversion (e.g. quantization types, ...)

    iou_threshold: float
        The IoU threshold

    conf_threshold: float
        The confidence threshold
    """

    def __init__(self, model_parameters, conversion_parameters, iou_threshold, conf_threshold):
        self.model_parameters = model_parameters
        self.conversion_parameters = conversion_parameters
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def generate(self):
        """ Generates a representative dataset from the source dataset

        Returns
        ----------
        representative_dataset
            The representative dataset that can be used for the TFlite conversion
        """
        source = self.__download_representative_dataset()
        dataset = LoadImages(source, img_size=self.model_parameters.img_size, auto=False)
        return self.__representative_dataset_gen(dataset)

    def __representative_dataset_gen(self, dataset):
        def representative_dataset():
            # Representative dataset for use with tf_converter.representative_dataset
            n = 0
            for path, img, im0s, vid_cap in dataset:
                # Get sample input data as a numpy array in a method of your choosing.
                n += 1
                input = np.transpose(img, [1, 2, 0])
                input = np.expand_dims(input, axis=0).astype(np.float32)
                if not self.model_parameters.include_normalization:
                    input /= 255.0
                if self.model_parameters.include_nms:
                    yield [input, np.array([self.iou_threshold], dtype=np.float32),
                           np.array([self.conf_threshold], dtype=np.float32)]
                else:
                    yield [input]
                if n >= self.conversion_parameters.nb_calib:
                    break

        return representative_dataset

    def __download_representative_dataset(self):
        # Creates target directory
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Get the name of the dataset and dataset zip
        file_path = get_zipfile_path(self.conversion_parameters.source)
        dataset_path = file_path.rstrip('.zip')
        # If it does not exist, download it
        if not os.path.exists(dataset_path):
            logging.info(f'Downloading reference dataset for {self.conversion_parameters.source}...')
            dataset_url = get_dataset_url(self.conversion_parameters.source)
            self.__safe_download(file_path, dataset_url, min_bytes=1E0,
                                 error_msg=f'Download fail: could not download the representative dataset at {dataset_url}')
            archive = zipfile.ZipFile(file_path)
            for file in archive.namelist():
                archive.extract(file, DATA_DIR)
            Path(file_path).unlink()
        return dataset_path

    @staticmethod
    def __safe_download(file, url, min_bytes=1E0, error_msg=''):
        # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
        file = Path(file)
        try:  # GitHub
            logging.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, str(file))
            assert file.exists() and file.stat().st_size > min_bytes  # check
        except Exception as e:  # GCP
            file.unlink(missing_ok=True)  # remove partial downloads
            raise Exception(f'ERROR: Download failure: {error_msg or url}')
        finally:
            if not file.exists() or file.stat().st_size < min_bytes:  # check
                file.unlink(missing_ok=True)  # remove partial downloads
