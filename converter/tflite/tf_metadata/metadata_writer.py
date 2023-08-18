import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from helpers.constants import FULLINT8, METADATA_FILE_NAME, SEGMENTATION, RED, END_COLOR, BLUE, GREEN
from tf_metadata.input_metadata_writer import InputMetadataWriter
from tf_metadata.output_metadata_writer import OutputMetadataWriter
from tf_utils.parameters import ModelParameters
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb


class MetadataWriter:
    """ Class that writes the metadata into a TFLite model

    Attributes
    ----------
    input_order: List[str]
        The names of the inputs in order

    output_order: List[str]
        The names of the outputs in order

    model_parameters: ModelParameters
        The parameters for the model to be converted (e.g. type, use nms, ...)

    quantization_type: str
        The quantization type (`float32`, `float16` or `int8`)

    tflite_path: str
        The path to the TFLite model in which to write the metadata
    """

    def __init__(self, input_order: List[str], output_order: List[str], model_parameters: ModelParameters,
                 quantization_type: str, tflite_path: str):

        self.tflite_path = tflite_path
        # Input
        self.model_type = model_parameters.model_type
        self.input_resolution = model_parameters.input_resolution
        self.include_normalization = model_parameters.include_normalization
        self.quantization_type = quantization_type
        # Output
        self.model_orig = model_parameters.model_orig
        self.include_nms = model_parameters.include_nms
        self.input_order = input_order
        self.output_order = output_order
        self.metadata = model_parameters.metadata
        self.max_det = model_parameters.max_det

    def write(self):
        """ Writes the metadata into the TFLite model """
        self.tmp_dir_path = Path(tempfile.mkdtemp())
        self.__create_associated_files()
        self.__create_model_info()

        try:
            logging.info(f'{BLUE}Populating metadata for the TFLite model...{END_COLOR}')
            input_writer = InputMetadataWriter(self.input_order, self.input_resolution, self.include_normalization,
                                               multiple_inputs=self.include_nms)
            self.input_meta = input_writer.write()

            output_writer = OutputMetadataWriter(self.output_order, self.metadata_path, len(self.metadata['names']),
                                                 self.max_det,
                                                 self.include_nms, self.model_orig, self.model_type)
            self.output_meta, self.output_group = output_writer.write()

            self.__create_subgraph()
            self.__populate_metadata()
            logging.info(f'{GREEN}TFLite metadata population success.{END_COLOR}')
        except Exception as e:
            raise ValueError(f'{RED}TFLite metadata population failure:{END_COLOR} {e}')
        finally:
            shutil.rmtree(self.tmp_dir_path)

    def __create_associated_files(self):
        # Create the labels file
        self.metadata_path = self.tmp_dir_path / METADATA_FILE_NAME
        self.metadata.update({'input_names': self.input_order, 'output_names': self.output_order, 'quantization_type': self.quantization_type})
        with self.metadata_path.open('w') as f:
            f.write(str(self.metadata))

    def __create_model_info(self):
        # Write model basic info
        self.model_meta = _metadata_fb.ModelMetadataT()
        self.model_meta.name = self.metadata['description']
        self.model_meta.version = self.metadata['version']
        self.model_meta.author = self.metadata['author']
        self.model_meta.license = self.metadata['license']
        if self.model_type == SEGMENTATION:
            self.model_meta.description = (
                "Detect object in a image and show them with bounding boxes and segmentation.")
        else:
            self.model_meta.description = (
                "Detect object in a image and show them with bounding boxes.")

    def __create_subgraph(self):
        # Creates subgraph with the input and output
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = self.input_meta
        subgraph.outputTensorMetadata = self.output_meta
        if self.output_group is not None:
            subgraph.outputTensorGroups = self.output_group
        self.model_meta.subgraphMetadata = [subgraph]

    def __populate_metadata(self):
        # Populates the metadata with the metadata and labels file
        b = flatbuffers.Builder(0)
        b.Finish(
            self.model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(self.tflite_path)
        populator.load_metadata_buffer(metadata_buf)

        # Add labels.txt to the tflite model.
        populator.load_associated_files([self.metadata_path])
        populator.populate()
