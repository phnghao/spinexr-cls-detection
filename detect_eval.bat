@echo off

echo eval yolov8

python -m spine.detection.eval ^
    --weights-path "./runs/detect/train/weights/best.pt" ^
    --data-yaml "./data/detection/yolo/test.yaml"