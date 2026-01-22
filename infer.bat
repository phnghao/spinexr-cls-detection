@echo off
python -m spine.inference ^
    --config-file "./spine/configs/sparsercnn.yaml" ^
    --det-model "./outputs/spine_sparsercnn/model_final.pth" ^
    --cls-model "./outputs/densenet/best.pth" ^
    --out-dir "./outputs/json_results"