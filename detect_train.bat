@echo off

echo train yolov8

yolo setting reset
python -m spine.detection.train_yolo ^
    --yaml-file "./data/detection/yolo/train.yaml" ^
    --epochs 120 ^
    --batch-size 16 ^
    --imgsz 640 ^
    --device 0 

REM train yolo with batch 32
REM python -m spine.detection.train_yolo ^
REM     --yaml-file "./data/detection/yolo/train.yaml" ^
REM     --epochs 239 ^
REM     --batch-size 16 ^
REM     --imgsz 640 ^
REM     --device 0
