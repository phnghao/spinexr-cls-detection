@echo off
echo Training
python -m spine.classification.train_net ^
    --csv-file "./data/classification/train.csv" ^
    --image-dir "./data/train_pngs" ^
    --save-file "./cls_save/train/best.pth" ^
    --nepochs 1 ^
    --batch-size 16 ^
    --num-workers 4 ^
    --image-size 224