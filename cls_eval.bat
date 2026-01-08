@echo off
python -m spine.classification.eval ^
  --csv-file "./data/classification/test.csv" ^
  --image-dir "./data/test_pngs" ^
  --model-path "./outputs/densenet201/best.pth" ^
  --batch-size 4
