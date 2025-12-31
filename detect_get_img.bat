@echo off
echo generating images
python -m spine.detection.gen_image ^
    --root-dir "./data/train_pngs" ^
    --traincsv-file "./data/detection/train.csv" ^
    --valcsv-file "./data/detection/val.csv" ^
    --train-dir "./data/detection/yolo/train/images" ^
    --val-dir "./data/detection/yolo/val/images" ^
    --cpus 4