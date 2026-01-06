@echo off

echo train yolov8

yolo setting reset
python -m spine.detection.train_yolo ^
     --yaml-file "./data/detection/yolo/train.yaml" ^
     --epochs 120 ^
     --batch-size 16 ^
     --imgsz 640 ^
     --device 0
