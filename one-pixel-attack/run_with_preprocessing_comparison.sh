#!/bin/bash

# Example script to run the evaluation with preprocessing comparison

# Set your paths
DATA_PATH="path/to/your/data.csv"
SOURCE_FOLDER="path/to/original/images"
OUTPUT_FOLDER="deid_png_onepix"
PREPROCESSED_FOLDER="deid_png_preprocessed"
RESULTS_FOLDER="./robustness_results"

# Run with preprocessing comparison
python run_robustness_evaluation.py \
    --data-path "$DATA_PATH" \
    --source-folder "$SOURCE_FOLDER" \
    --output-folder "$OUTPUT_FOLDER" \
    --preprocessed-folder "$PREPROCESSED_FOLDER" \
    --results-folder "$RESULTS_FOLDER" \
    --include-preprocessed-eval \
    --pathology "Pneumonia" \
    --max-iter 100 \
    --pop-size 200 \
    --limit 10  # Remove this to process all samples

# To skip preprocessing generation if already done:
# python run_robustness_evaluation.py \
#     --data-path "$DATA_PATH" \
#     --source-folder "$SOURCE_FOLDER" \
#     --output-folder "$OUTPUT_FOLDER" \
#     --preprocessed-folder "$PREPROCESSED_FOLDER" \
#     --results-folder "$RESULTS_FOLDER" \
#     --include-preprocessed-eval \
#     --skip-preprocessing \
#     --skip-attack-generation \
#     --pathology "Pneumonia" \
#     --max-iter 100 \
#     --pop-size 200