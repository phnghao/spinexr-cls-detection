@echo off

echo train yolov8

python -m spine.detection.train_yolo ^
    --yaml-file "./data/detection/yolo/train.yaml" ^
    --epochs 124 ^
    --batch-size 16 ^
    --imgsz 640 ^
    --device 0 