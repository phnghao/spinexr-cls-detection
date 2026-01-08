@echo off

echo training sparse rcnn
python -m spine.sparsercnn.train ^
    --config-file "./spine/configs/sparsercnn.res101.300pro.3x.yaml" ^
    --num-gpus 1