@echo off

echo training sparse rcnn
python -m spine.sparsercnn.train ^
    --config-file "./spine/configs/sparsercnn.yaml" ^
    --num-gpus 1