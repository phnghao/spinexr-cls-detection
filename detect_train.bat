@echo off

echo train Yolov8

python -m spine.detection.train_yolo ^
    --yaml-file "./data/detection/yolo/data.yaml" ^
    --epochs 1 ^
    --batch-size 4 ^
    --imgsz 640 ^
    --device 0 