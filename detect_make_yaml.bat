@echo off
echo Created train.yaml

set ROOT_DIR=./data/detection/yolo
set TRAIN_IMAGES=train/images
set VAL_IMAGES=val/images
set OUTPUT=train.yaml

python -m spine.detection.make_train_yaml ^
    --root-dir %ROOT_DIR% ^
    --train-images %TRAIN_IMAGES% ^
    --val-images %VAL_IMAGES% ^
    --output %OUTPUT%

echo Created test.yaml
python -m spine.detection.make_test_yaml ^
    --root-dir "./data/detection/yolo/test" ^
    --train-images "images" ^
    --val-images "images" ^
    --test-images "images" ^
    --output "./data/detection/yolo/test.yaml"

