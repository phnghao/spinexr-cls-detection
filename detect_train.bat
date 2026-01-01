@echo off

echo train yolov8

python -m spine.detection.train_yolo ^
    --yaml-file "./data/detection/yolo/train.yaml" ^
    --epochs 1 ^
    --batch-size 4 ^
    --imgsz 640 ^
    --device 0 