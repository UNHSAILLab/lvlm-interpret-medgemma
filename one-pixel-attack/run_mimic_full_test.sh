#!/bin/bash

# Full MIMIC CXR Preprocessing Effect Test
# This script tests the preprocessing effect on all 100 MIMIC samples with different pathologies

echo "==================================================="
echo "MIMIC CXR Full Dataset Preprocessing Effect Test"
echo "==================================================="
echo ""

# First, convert MIMIC data format if not already done
echo "Step 1: Preparing MIMIC data format..."
python -c "
from mimic_data_adapter import adapt_mimic_to_pipeline_format
adapt_mimic_to_pipeline_format(
    '/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample.csv',
    './mimic_adapted_questions.csv'
)
print('Data conversion complete.')
"

echo ""
echo "Step 2: Running tests for different pathologies..."
echo ""

# Array of pathologies to test
# Based on the model's available pathologies:
# Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, 
# Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening,
# Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture,
# Lung Opacity, Enlarged Cardiomediastinum

pathologies=("Effusion" "Pneumonia" "Atelectasis" "Consolidation" "Edema")

# Create a results summary directory
SUMMARY_DIR="./mimic_robustness_results/summary_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUMMARY_DIR"

# Test each pathology
for pathology in "${pathologies[@]}"; do
    echo "================================================"
    echo "Testing pathology: $pathology"
    echo "================================================"
    
    # Create pathology-specific output folders
    ATTACK_FOLDER="/home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack_${pathology,,}"
    PREPROCESSED_FOLDER="/home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only"
    RESULTS_FOLDER="./mimic_robustness_results/${pathology,,}_results"
    
    # Run the evaluation
    python run_robustness_evaluation.py \
        --data-path ./mimic_adapted_questions.csv \
        --source-folder /home/bsada1/mimic_cxr_hundred_vqa \
        --output-folder "$ATTACK_FOLDER" \
        --preprocessed-folder "$PREPROCESSED_FOLDER" \
        --results-folder "$RESULTS_FOLDER" \
        --include-preprocessed-eval \
        --pathology "$pathology" \
        --max-iter 10 \
        --pop-size 20 \
        --skip-preprocessing
    
    # Copy the summary to the combined results folder
    latest_run=$(ls -t "$RESULTS_FOLDER" | head -1)
    if [ -f "$RESULTS_FOLDER/$latest_run/robustness_summary.json" ]; then
        cp "$RESULTS_FOLDER/$latest_run/robustness_summary.json" \
           "$SUMMARY_DIR/${pathology,,}_summary.json"
    fi
    
    echo ""
    echo "Completed testing for $pathology"
    echo ""
done

echo "================================================"
echo "Step 3: Creating combined summary report..."
echo "================================================"

# Create a Python script to generate summary report
python - <<EOF
import json
import pandas as pd
from pathlib import Path

summary_dir = Path("$SUMMARY_DIR")
pathologies = $( printf '%s\n' "${pathologies[@]}" | jq -R . | jq -s . )

results = []
for pathology in pathologies:
    summary_file = summary_dir / f"{pathology.lower()}_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            data['pathology'] = pathology
            results.append(data)

if results:
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Calculate averages
    avg_row = {
        'pathology': 'AVERAGE',
        'total_samples': df['total_samples'].mean(),
        'baseline_accuracy': df['baseline_accuracy'].mean(),
        'preprocessed_accuracy': df['preprocessed_accuracy'].mean(),
        'attack_accuracy': df['attack_accuracy'].mean(),
        'preprocessing_effect': df['preprocessing_effect'].mean(),
        'attack_effect': df['attack_effect'].mean(),
        'attack_success_rate': df['attack_success_rate'].mean(),
        'targeted_success_rate': df['targeted_success_rate'].mean(),
    }
    
    # Add average row
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(summary_dir / 'combined_results.csv', index=False)
    
    # Print summary
    print("\nCOMBINED RESULTS SUMMARY")
    print("========================")
    print(df[['pathology', 'baseline_accuracy', 'preprocessed_accuracy', 
             'attack_accuracy', 'preprocessing_effect', 'attack_effect']].to_string(index=False))
    
    # Save markdown report
    with open(summary_dir / 'report.md', 'w') as f:
        f.write("# MIMIC CXR Preprocessing Effect Analysis\n\n")
        f.write("## Summary Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        f.write(f"- Average Baseline Accuracy: {avg_row['baseline_accuracy']:.1%}\n")
        f.write(f"- Average Preprocessing Effect: {avg_row['preprocessing_effect']:+.1%}\n")
        f.write(f"- Average Attack Effect: {avg_row['attack_effect']:+.1%}\n")
        f.write(f"- Average Attack Success Rate: {avg_row['attack_success_rate']:.1%}\n")
        
    print(f"\nResults saved to: {summary_dir}")
else:
    print("No results found!")
EOF

echo ""
echo "================================================"
echo "All tests complete!"
echo "Results saved to: $SUMMARY_DIR"
echo "================================================"