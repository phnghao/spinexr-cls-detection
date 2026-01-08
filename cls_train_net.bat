@echo off
echo Training
python -m spine.classification.train_net ^
    --csv-file "./data/classification/train.csv" ^
    --image-dir "./data/train_pngs" ^
    --save-file "./outputs/densenet201/best.pth" ^
    --nepochs 45 ^
    --batch-size 32 ^
    --num-workers 4 ^
    --image-size 224