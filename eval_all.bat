@echo off
echo evaluating all stage
python -m spine.eval_all ^
  --eval-task all ^
  --cls-csv "./data/classification/test.csv" ^
  --img-dir "./data/test_pngs" ^
  --cls-model "./outputs/densenet/best.pth" ^
  --cls-output "./outputs/cls_eval" ^
  --config-file "./spine/configs/sparsercnn.yaml" ^
  MODEL.WEIGHTS ./outputs/spine_sparsercnn/model_final.pth