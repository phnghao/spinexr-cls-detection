@echo off

echo split data
python -m spine.detection.split_data ^
    --csv-file "./data/train_prepro_data.csv" ^
    --train-output "./data/detection/train.csv" ^
    --val-output "./data/detection/val.csv" 