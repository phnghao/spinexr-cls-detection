@echo off
echo Convert train images to png
python spine/preprocess/dicom2png.py ^
    --input-dir "./data/dicom_dataset/train_images" ^
    --output-dir "./data/train_pngs" ^
    --cpus 8 ^
    --log-file "./data/cvtTrain_log.txt"

echo Convert test images to png
python spine/preprocess/dicom2png.py ^
    --input-dir "./data/dicom_dataset/test_images" ^
    --output-dir "./data/test_pngs" ^
    --cpus 8 ^
    --log-file "./data/cvtTest_log.txt"
