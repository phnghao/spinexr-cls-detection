@echo off
python -m spine.infer ^
    --cls-path "./cls_save/train/best.pth" ^
    --det-path "./runs/detect/train/weights/best.pt" ^
    --input-dir "./data/test_pngs" ^
    --output-dir "./outputs" ^
    --gt-csv "./data/test_prepro_data.csv" ^
    --cls-thr 0.28 ^
    --det-thr 0.28