@echo off


echo convert to train coco form
python -m spine.sparsercnn.convert_coco ^
  --csv-path ./data/train_prepro_data.csv ^
  --img-dir ./data/train_pngs ^
  --output-path ./data/annotations/train.json

