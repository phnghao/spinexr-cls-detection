@echo off

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