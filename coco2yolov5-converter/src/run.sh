export WANDB_API_KEY=6d2742d7371e881427099b365b37d7b3c67bc36c
python src/sbb2yolo.py --coco_input_folder /data --yolo_output_folder /output --dataset_split_pivot=0.9
python yolov5/train.py --epochs 100 --data /output/config.yaml --weights yolov5m.pt