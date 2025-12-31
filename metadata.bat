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