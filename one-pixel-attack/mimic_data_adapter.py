"""Adapter to make MIMIC CXR VQA data compatible with the robustness evaluation pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import ast


def adapt_mimic_to_pipeline_format(mimic_csv_path: str, output_csv_path: str = None):
    """
    Convert MIMIC CXR VQA format to the format expected by run_robustness_evaluation.py
    
    MIMIC format:
    - dicom_id, subject_id, study_id, question, question_type, answer, split, image_path
    
    Pipeline format needs:
    - study_id (use dicom_id), question, options, correct_answer, image_path, PatientID, ImagePath
    """
    
    # Read MIMIC data
    df = pd.read_csv(mimic_csv_path)
    
    # Create adapted dataframe
    adapted_df = pd.DataFrame()
    
    # Use dicom_id as study_id (unique identifier)
    adapted_df['study_id'] = df['dicom_id']
    
    # Copy question directly
    adapted_df['question'] = df['question']
    
    # For MIMIC yes/no questions, create options
    # Assuming most are yes/no based on the sample data
    adapted_df['options'] = df.apply(lambda row: ['yes', 'no', '', ''], axis=1)
    
    # Keep the answer as-is (yes/no) instead of mapping to A/B
    # Since MedGemma outputs yes/no directly
    adapted_df['correct_answer'] = df['answer'].str.lower().str.strip()
    
    # Format image_path as list (pipeline expects list format)
    adapted_df['image_path'] = df['image_path'].apply(lambda x: [x])
    
    # Add PatientID (using subject_id)
    adapted_df['PatientID'] = df['subject_id']
    
    # Add ImagePath (same as image_path but as string)
    adapted_df['ImagePath'] = df['image_path']
    
    # Add any other required columns with defaults
    adapted_df['ViewPosition'] = 'PA'  # Default view
    
    # Save if output path provided
    if output_csv_path:
        adapted_df.to_csv(output_csv_path, index=False)
        print(f"Adapted data saved to: {output_csv_path}")
    
    return adapted_df


def create_mimic_run_script():
    """Generate a bash script to run the evaluation on MIMIC data."""
    
    script_content = """#!/bin/bash

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

python run_robustness_evaluation.py \\
    --data-path ./mimic_adapted_questions.csv \\
    --source-folder /home/bsada1/mimic_cxr_hundred_vqa \\
    --output-folder /home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack \\
    --preprocessed-folder /home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only \\
    --results-folder ./mimic_robustness_results \\
    --include-preprocessed-eval \\
    --pathology "Pleural Effusion" \\
    --max-iter 100 \\
    --pop-size 200

echo "Analysis complete. Results saved to ./mimic_robustness_results/"
"""
    
    script_path = "/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/run_mimic_test.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


if __name__ == "__main__":
    # Test the adapter
    print("Testing MIMIC data adapter...")
    
    # Convert MIMIC format
    mimic_path = "/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample.csv"
    output_path = "/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv"
    
    adapted_df = adapt_mimic_to_pipeline_format(mimic_path, output_path)
    
    print(f"\nAdapted {len(adapted_df)} questions")
    print("\nFirst few rows of adapted data:")
    print(adapted_df.head())
    
    # Create run script
    script_path = create_mimic_run_script()
    print(f"\nRun script created at: {script_path}")
    print("\nTo run the test, execute:")
    print(f"chmod +x {script_path}")
    print(f"bash {script_path}")