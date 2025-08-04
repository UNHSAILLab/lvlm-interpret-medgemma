#!/bin/bash

# Fast MIMIC CXR Test - Skip preprocessing, use fast attack parameters
# This version is optimized for speed to quickly test all 100 samples

echo "==================================================="
echo "MIMIC CXR Fast Full Dataset Test"
echo "==================================================="
echo ""

# Ensure data is converted
python -c "
from mimic_data_adapter import adapt_mimic_to_pipeline_format
adapt_mimic_to_pipeline_format(
    '/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample.csv',
    './mimic_adapted_questions.csv'
)
"

# Single run with Effusion pathology (most relevant for pleural effusion questions)
echo "Running fast test on all 100 MIMIC samples..."
echo "Pathology: Effusion (most relevant for dataset)"
echo "Attack parameters: 5 iterations, 10 population (very fast)"
echo ""

python run_robustness_evaluation.py \
    --data-path ./mimic_adapted_questions.csv \
    --source-folder /home/bsada1/mimic_cxr_hundred_vqa \
    --output-folder /home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack \
    --preprocessed-folder /home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only \
    --results-folder ./mimic_robustness_results \
    --include-preprocessed-eval \
    --pathology "Effusion" \
    --max-iter 5 \
    --pop-size 10 \
    --skip-preprocessing

echo ""
echo "Test complete! Check ./mimic_robustness_results for results."