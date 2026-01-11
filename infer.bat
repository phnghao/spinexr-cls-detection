@echo off
python -m spine.infer_vis ^
    --config-file "./spine/configs/sparsercnn.yaml" ^
    --det-model "./outputs/spine_sparsercnn/model_0049999.pth" ^
    --cls-model "./outputs/densenet/best.pth" ^
    -- num-samples 20