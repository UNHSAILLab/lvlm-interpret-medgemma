#!/bin/bash

python run_robustness_evaluation.py \
  --data-path /home/bsada1/ReXGradient-160K/to_attack.csv \
  --source-folder /home/bsada1/ReXGradient-160K \
  --output-folder /home/bsada1/ReXGradient-160K/deid_png_onepx_pneumo_2 \
  --include-preprocessed-eval \
  --preprocessed-folder /home/bsada1/ReXGradient-160K/deid_png_onepx_preproc \
  --results-folder ./robustness_results \
  --limit 100 \
  --use-database


    python run_robustness_evaluation.py \
    --data-path ./mimic_adapted_questions.csv \
    --source-folder /home/bsada1/mimic_cxr_hundred_vqa \
    --output-folder /home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack \
    --preprocessed-folder /home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only \
    --results-folder ./mimic_robustness_results \
    --include-preprocessed-eval \
    --limit 2