import json
import random
import shutil
from pathlib import Path

import pytest
import yaml

from src.sbb2yolo import create_parser, SBB2Yolo


def create_dummy_dataset(tmp_dir, name):
    dataset_path = tmp_dir / Path(name)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset_path.mkdir()

    nb_class = random.randint(1, 3)
    classes = [{"id": i, "name": f"{name}_class{i}"} for i in range(nb_class)]
    files = [{"id": i, "file_name": f"{name}_file{i}.jpg"} for i in range(random.randint(1, 3))]

    id = 0
    bboxes = []
    annotations = {}
    # One object of each category per image
    for i, file in enumerate(files):
        f = open(dataset_path / Path(file["file_name"]), 'w')
        f.close()
        nb_a = random.randint(1, 3)
        annotations[file["file_name"]] = nb_a
        for _ in range(nb_a):
            class_id = random.randint(0, nb_class - 1)
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


@pytest.fixture(scope="session", autouse=True)
def initialize_dummy_datasets():
    # Create dummy coco datasets
    datasets = [f"dataset{i}" for i in range(1, 4)]
    tmp_dir = Path('./test_tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    exp_classes = []
    exp_images = []
    exp_annotations = {}
    for d in datasets:
        classes, images, annotations = create_dummy_dataset(tmp_dir, d)
        exp_classes.extend(classes)
        exp_images.extend(images)
        exp_annotations.update(annotations)

    # Converts the coco datasets into one big Yolo dataset
    datasets_path = [str(tmp_dir / Path(d)) for d in datasets]
    SBB2Yolo(coco_input_folders=datasets_path, yolo_output_folder=str(tmp_dir / "yolo"), output_yolo_config_file='config.yaml',
             dataset_split_pivot=random.uniform(0, 1)).convert()
    return tmp_dir / "yolo", exp_classes, exp_images, exp_annotations


class TestMain:
    def test_parser_works(self):
        parser = create_parser()
        arguments = parser.parse_args(
            ['--coco-input-folders', '/home/unittest/folder'])
        assert arguments.coco_input_folders[0] == '/home/unittest/folder'
        assert arguments.yolo_output_folder == 'output'
        assert arguments.yolo_config_file == 'config.yaml'
        assert arguments.dataset_split_pivot == 0.8

    def test_all_classes_present(self, initialize_dummy_datasets):
        tmp_dir, exp_classes, _, _ = initialize_dummy_datasets
        # Check all classes are in resulting dataset
        with open(tmp_dir / Path('config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        classes = config.get('names', [])
        assert len(classes) != 0
        assert len(classes) == len(exp_classes)
        for c in exp_classes:
            assert c in classes

    def test_all_images_present(self, initialize_dummy_datasets):
        tmp_dir, _, exp_images, _ = initialize_dummy_datasets
        # Check all images are in resulting dataset
        images = [f.name for f in (tmp_dir / Path("images")).glob("*/*.jpg")]
        assert len(images) != 0
        assert len(images) == len(exp_images)
        for f in exp_images:
            assert f in images

    def test_all_annotations_present(self, initialize_dummy_datasets):
        tmp_dir, _, _, exp_annotations = initialize_dummy_datasets
        # Check all images got the right number of annotations
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
                annotations = h.readlines()
            assert len(annotations) == exp_annotations[f]
