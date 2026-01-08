@echo off

echo Extract information of train DICOM images to csv
python spine/preprocess/metadata.py ^
    --input-dir "./data/dicom_dataset/train_images" ^
    --output-dir "./data/train_metadata.csv" ^
    --cpus 8 

echo Extract information of test DICOM images to csv
python spine/preprocess/metadata.py ^
    --input-dir "./data/dicom_dataset/test_images" ^
    --output-dir "./data/test_metadata.csv" ^
    --cpus 8

echo merge train's annotation and metadata
python spine/preprocess/merge_csv.py ^
    --annotation-file "./data/dicom_dataset/annotations/train.csv" ^
    --meta-file "./data/train_metadata.csv" ^
    --output-file "./data/train_prepro_data.csv"
echo merge test's annotation and metadata
python spine/preprocess/merge_csv.py ^
    --annotation-file "./data/dicom_dataset/annotations/test.csv" ^
    --meta-file "./data/test_metadata.csv" ^
    --output-file "./data/test_prepro_data.csv"

echo created classification train csv file
python spine/classification/cls_gencsv.py ^
  --input-csv "./data/train_prepro_data.csv" ^
  --output-csv "./data/classification/train.csv"

echo created classification test csv file
python spine/classification/cls_gencsv.py ^
  --input-csv "./data/test_prepro_data.csv" ^
  --output-csv "./data/classification/test.csv"

echo convert to train coco form
python -m spine.sparsercnn.convert_coco ^
  --csv-path ./data/train_prepro_data.csv ^
  --img-dir ./data/train_pngs ^
  --output-path ./data/annotations/train.json

echo convert to test coco form
python -m spine.sparsercnn.convert_coco ^
  --csv-path ./data/test_prepro_data.csv ^
  --img-dir ./data/test_pngs ^
  --output-path ./data/annotations/test.json