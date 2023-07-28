import json
import pytest
import random
import shutil
import yaml
from pathlib import Path

from src.sbb2yolo import create_parser, Coco2YoloDatasetConverter


def create_dummy_detection_dataset(tmp_dir, name):
    dataset_path = tmp_dir / Path(name)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.mkdir()

    nb_class = random.randint(1, 5)
    classes = [{"id": i, "name": f"{name}_class{i}"} for i in range(nb_class)]
    files = [{"id": i, "file_name": f"{name}_file{i}.jpg"} for i in range(random.randint(1, 3))]

    id = 0
    bboxes = []
    annotations = {}
    # One to three annotations per image
    for i, file in enumerate(files):
        f = open(dataset_path / Path(file["file_name"]), 'w')
        f.close()
        nb_a = random.randint(1, 5)
        annotations[file["file_name"]] = []
        for _ in range(nb_a):
            class_id = random.randint(0, nb_class - 1)
            annotations[file["file_name"]].append(class_id)
            bboxes.append(
                {"id": id, "image_id": i, "category_id": class_id, "bbox": [random.uniform(0.1, 1) for _ in range(4)]})
            id += 1

    cocojson = {}
    cocojson["categories"] = classes
    cocojson["images"] = files
    cocojson["annotations"] = bboxes
    with open(dataset_path / Path('coco.json'), 'w') as f:
        json.dump(cocojson, f)
    return [c["name"] for c in classes], [f["file_name"] for f in files], annotations


def create_dummy_segmentation_dataset(tmp_dir, name):
    dataset_path = tmp_dir / Path(name)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.mkdir()

    nb_class = random.randint(1, 5)
    classes = [{"id": i, "name": f"{name}_class{i}"} for i in range(nb_class)]
    files = [{"id": i, "file_name": f"{name}_file{i}.jpg"} for i in range(random.randint(1, 3))]

    id = 0
    masks = []
    annotations = {}
    # One mask per image
    for i, file in enumerate(files):
        f = open(dataset_path / Path(file["file_name"]), 'w')
        f.close()
        class_id = random.randint(0, nb_class - 1)
        annotations[file["file_name"]] = class_id
        masks.append(
            {"id": id, "image_id": i, "category_id": class_id,
             "segmentation": [[random.uniform(0.1, 1) for _ in range(10)]]})
        id += 1

    cocojson = {}
    cocojson["categories"] = classes
    cocojson["images"] = files
    cocojson["annotations"] = masks
    with open(dataset_path / Path('coco.json'), 'w') as f:
        json.dump(cocojson, f)
    return [c["name"] for c in classes], [f["file_name"] for f in files], annotations


@pytest.fixture(scope="session", autouse=True)
def initialize_dummy_datasets():
    # Create dummy coco datasets
    datasets = [f"dataset{i}" for i in range(1, 2)]
    tmp_dir = Path('./test_tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    exp_detection_classes = []
    exp_detection_images = []
    exp_detection_annotations = {}
    exp_segmentation_classes = []
    exp_segmentation_images = []
    exp_segmentation_annotations = {}
    for d in datasets:
        classes_detection, images_detection, detection_annotations = create_dummy_detection_dataset(tmp_dir,
                                                                                                    f"{d}_detection")
        classes_segmentation, images_segmentation, segmentation_annotations = create_dummy_segmentation_dataset(tmp_dir,
                                                                                                                f"{d}_segmentation")
        exp_detection_classes.extend(classes_detection)
        exp_detection_images.extend(images_detection)
        exp_detection_annotations.update(detection_annotations)
        exp_segmentation_classes.extend(classes_segmentation)
        exp_segmentation_images.extend(images_segmentation)
        exp_segmentation_annotations.update(segmentation_annotations)

    # Converts the coco datasets into one big Yolo dataset
    detection_datasets_path = [str(tmp_dir / Path(f"{d}_detection")) for d in datasets]
    segmentation_datasets_path = [str(tmp_dir / Path(f"{d}_segmentation")) for d in datasets]
    Coco2YoloDatasetConverter(coco_input_folders=detection_datasets_path,
                              yolo_output_folder=str(tmp_dir / "yolo_detection"),
                              yolo_config_file='config_detection.yaml',
                              dataset_split_pivot=random.uniform(0, 1),
                              new_format=False).convert()
    Coco2YoloDatasetConverter(coco_input_folders=segmentation_datasets_path,
                              yolo_output_folder=str(tmp_dir / "yolo_segmentation"),
                              yolo_config_file='config_segmentation.yaml',
                              dataset_split_pivot=random.uniform(0, 1),
                              new_format=False).convert()
    return tmp_dir / "yolo_detection", tmp_dir / "yolo_segmentation", exp_detection_classes, exp_detection_images, exp_detection_annotations, exp_segmentation_classes, exp_segmentation_images, exp_segmentation_annotations


class TestMain:
    def test_parser_works(self):
        parser = create_parser()
        arguments = parser.parse_args(
            ['--coco-input-folders', '/home/unittest/folder'])
        assert arguments.coco_input_folders[0] == '/home/unittest/folder'
        assert arguments.yolo_output_folder == 'output'
        assert arguments.yolo_config_file == 'config.yaml'
        assert arguments.dataset_split_pivot == 0.8

    def test_all_classes_present_detection(self, initialize_dummy_datasets):
        tmp_dir, _, exp_classes, _, _, _, _, _ = initialize_dummy_datasets
        # Check all classes are in resulting dataset
        with open(tmp_dir / Path('config_detection.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        classes = config.get('names', [])
        assert len(classes) != 0
        assert len(classes) == len(exp_classes)
        for c in exp_classes:
            assert c in classes

    def test_all_classes_present_segmentation(self, initialize_dummy_datasets):
        _, tmp_dir, _, _, _, exp_classes, _, _ = initialize_dummy_datasets
        # Check all classes are in resulting dataset
        with open(tmp_dir / Path('config_segmentation.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        classes = config.get('names', [])
        assert len(classes) != 0
        assert len(classes) == len(exp_classes)
        for c in exp_classes:
            assert c in classes

    def test_all_images_present_detection(self, initialize_dummy_datasets):
        tmp_dir, _, _, exp_images, _, _, _, _ = initialize_dummy_datasets
        # Check all images are in resulting dataset
        images = [f.name for f in (tmp_dir / Path("images")).glob("*/*.jpg")]
        assert len(images) != 0
        assert len(images) == len(exp_images)
        for f in exp_images:
            assert f in images

    def test_all_images_present_segmentation(self, initialize_dummy_datasets):
        _, tmp_dir, _, _, _, _, exp_images, _ = initialize_dummy_datasets
        # Check all images are in resulting dataset
        images = [f.name for f in (tmp_dir / Path("images")).glob("*/*.jpg")]
        assert len(images) != 0
        assert len(images) == len(exp_images)
        for f in exp_images:
            assert f in images

    def test_all_annotations_present_detection(self, initialize_dummy_datasets):
        tmp_dir, _, _, _, exp_annotations, _, _, _ = initialize_dummy_datasets
        # Check all images got the right annotations (by class id)
        images = [f.name for f in (tmp_dir / Path("images")).glob("*/*.jpg")]
        for f in images:
            annotation = Path(f).with_suffix(".txt")
            train_path = (tmp_dir / Path("labels/train") / annotation)
            val_path = (tmp_dir / Path("labels/val") / annotation)
            assert train_path.exists() or val_path.exists()
            if train_path.exists():
                label_path = tmp_dir / Path("labels/train") / annotation
            else:
                label_path = tmp_dir / Path("labels/val") / annotation
            with label_path.open() as h:
                annotations = [int(i.split(' ')[0]) for i in h.readlines()]
            assert annotations == exp_annotations[f]

    def test_all_annotations_present_segmentation(self, initialize_dummy_datasets):
        _, tmp_dir, _, _, _, _, _, exp_annotations = initialize_dummy_datasets
        # Check all images got the right annotations (by class id)
        images = [f.name for f in (tmp_dir / Path("images")).glob("*/*.jpg")]
        for f in images:
            annotation = Path(f).with_suffix(".txt")
            train_path = (tmp_dir / Path("labels/train") / annotation)
            val_path = (tmp_dir / Path("labels/val") / annotation)
            assert train_path.exists() or val_path.exists()
            if train_path.exists():
                label_path = tmp_dir / Path("labels/train") / annotation
            else:
                label_path = tmp_dir / Path("labels/val") / annotation
            with label_path.open() as h:
                annotations = [int(i.split(' ')[0]) for i in h.readlines()]
            assert len(annotations) == 1
            assert annotations[0] == exp_annotations[f]
