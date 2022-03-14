export WANDB_API_KEY=insertkey
python src/sbb2yolo.py --coco_input_folder /data --yolo_output_folder /output --dataset_split_pivot=0.9
python yolov5/train.py --epochs 100 --data /output/config.yaml --weights yolov5m.pt