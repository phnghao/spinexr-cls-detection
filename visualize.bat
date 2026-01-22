@echo off

echo Display random images

python -m spine.visualize_res ^
    --vis-json "./outputs/json_results/visualize_results.json" ^
    --pred-json "./outputs/json_results/predicted_results.json" ^
    --out-dir "./outputs" ^
    --num-samples 2077

pause
