@echo off
echo Created data.yaml

set ROOT_DIR=./data/detection/yolo
set TRAIN_IMAGES=train/images
set VAL_IMAGES=val/images
set OUTPUT=data.yaml

python -m spine.detection.make_yaml ^
    --root-dir %ROOT_DIR% ^
    --train-images %TRAIN_IMAGES% ^
    --val-images %VAL_IMAGES% ^
    --output %OUTPUT%

