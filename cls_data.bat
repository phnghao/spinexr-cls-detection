@echo off
echo created classification train csv file
python spine/classification/cls_gencsv.py ^
  --input-csv "./data/train_prepro_data.csv" ^
  --output-csv "./data/classification/train.csv"

echo created classification test csv file
python spine/classification/cls_gencsv.py ^
  --input-csv "./data/test_prepro_data.csv" ^
  --output-csv "./data/classification/test.csv"