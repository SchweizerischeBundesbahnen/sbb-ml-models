import shutil
import tempfile
from pathlib import Path

from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

from constants import FULLINT8, LABELS_NAME
from tf_metadata.input_metadata_writer import InputMetadataWriter
from tf_metadata.output_metadata_writer import OutputMetadataWriter


class MetadataWriter:
    def __init__(self, input_order, output_order, model_parameters, labels, max_det,
                 quantized, tflite_path):

        self.tflite_path = tflite_path
        # Input
        self.include_threshold = model_parameters.include_threshold
        self.img_size = model_parameters.img_size
        self.include_normalization = model_parameters.include_normalization
        self.quantized = quantized
        # Output
        self.include_nms = model_parameters.include_nms
        self.input_order = input_order
        self.output_order = output_order
        self.labels = labels
        self.max_det = max_det

    def write(self):
        self.tmp_dir_path = Path(tempfile.mkdtemp())
        self.__create_associated_files()
        self.__create_model_info()

        try:
            input_writer = InputMetadataWriter(self.input_order, self.img_size, self.include_normalization,
                                               quantized=self.quantized == FULLINT8 and not self.include_nms,
                                               multiple_inputs=not self.include_threshold and self.include_nms)
            self.input_meta = input_writer.write()

            output_writer = OutputMetadataWriter(self.output_order, self.labels_path, len(self.labels), self.max_det,
                                                 self.include_nms)
            self.output_meta, self.output_group = output_writer.write()

            self.__create_subgraph()
            self.__populate_metadata()
        except Exception as e:
            raise ValueError(f"ERROR: {e}")
        finally:
            shutil.rmtree(self.tmp_dir_path)

    def __create_associated_files(self):
        # Labels file
        self.labels_path = self.tmp_dir_path / LABELS_NAME
        with self.labels_path.open('w') as h:
            for l in self.labels:
                h.write(f'{l}\n')

    def __create_model_info(self):
        self.model_meta = _metadata_fb.ModelMetadataT()
        self.model_meta.name = "YoloV5 object detector"
        self.model_meta.description = ("Detect object in a image and show them with bounding boxes.")
        self.model_meta.version = "v1"

    def __create_subgraph(self):
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = self.input_meta
        subgraph.outputTensorMetadata = self.output_meta
        if self.output_group is not None:
            subgraph.outputTensorGroups = self.output_group
        self.model_meta.subgraphMetadata = [subgraph]

    def __populate_metadata(self):
        b = flatbuffers.Builder(0)
        b.Finish(
            self.model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(self.tflite_path)
        populator.load_metadata_buffer(metadata_buf)

        # Add labels.txt to the tflite model.
        populator.load_associated_files([self.labels_path])
        populator.populate()
