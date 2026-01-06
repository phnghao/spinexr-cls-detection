@echo off

echo train yolov8

yolo setting reset
python -m spine.detection.train_yolo ^
     --yaml-file "./data/detection/yolo/train.yaml" ^
     --epochs 239 ^
     --batch-size 32 ^
     --imgsz 640 ^
     --device 0
