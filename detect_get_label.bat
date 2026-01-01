@echo off

echo train label
python -m spine.detection.gen_label ^
    --anno-file "./data/detection/train.csv" ^
    --output-dir "./data/detection/yolo/train/labels" ^
    --cpus 4

python -m spine.detection.gen_label ^
    --anno-file "./data/detection/val.csv" ^
    --output-dir "./data/detection/yolo/val/labels" ^
    --cpus 4

python -m spine.detection.gen_label ^
    --anno-file "./data/test_prepro_data.csv" ^
    --output-dir "./data/detection/yolo/test/labels" ^
    --cpus 4