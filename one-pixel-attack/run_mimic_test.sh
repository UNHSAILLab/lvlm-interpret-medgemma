#!/bin/bash

# MIMIC CXR Preprocessing Effect Test
# This script tests whether the preprocessing pipeline improves performance on MIMIC data

# Step 1: Convert MIMIC data to pipeline format
echo "Converting MIMIC data format..."
python -c "
from mimic_data_adapter import adapt_mimic_to_pipeline_format
adapt_mimic_to_pipeline_format(
    '/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample.csv',
    '/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv'
)
"

# Step 2: Run the three-way comparison
echo "Running three-way comparison on MIMIC data..."

python run_robustness_evaluation.py \
    --data-path ./mimic_adapted_questions.csv \
    --source-folder /home/bsada1/mimic_cxr_hundred_vqa \
    --output-folder /home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack \
    --preprocessed-folder /home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only \
    --results-folder ./mimic_robustness_results \
    --include-preprocessed-eval \
    --pathology "Pleural Effusion" \
    --max-iter 100 \
    --pop-size 200

echo "Analysis complete. Results saved to ./mimic_robustness_results/"
