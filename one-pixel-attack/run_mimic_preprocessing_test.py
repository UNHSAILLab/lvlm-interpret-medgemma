#!/usr/bin/env python
"""
Quick script to test preprocessing effect on MIMIC CXR data
"""

import subprocess
import sys
from pathlib import Path

def main():
    # First ensure we have the adapted CSV
    adapter_script = Path(__file__).parent / "mimic_data_adapter.py"
    print("Step 1: Preparing MIMIC data...")
    subprocess.run([sys.executable, str(adapter_script)], check=True)
    
    # Now run the evaluation
    print("\nStep 2: Running preprocessing effect test on MIMIC data...")
    
    cmd = [
        sys.executable,
        "run_robustness_evaluation.py",
        "--data-path", "./mimic_adapted_questions.csv",
        "--source-folder", "/home/bsada1/mimic_cxr_hundred_vqa",
        "--output-folder", "/home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack",
        "--preprocessed-folder", "/home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only",
        "--results-folder", "./mimic_robustness_results",
        "--include-preprocessed-eval",
        "--pathology", "Pleural Effusion",
        "--max-iter", "100",
        "--pop-size", "200",
        # Optional: limit samples for quick test
        # "--limit", "10"
    ]
    
    subprocess.run(cmd, check=True)
    
    print("\nTest complete! Results saved to ./mimic_robustness_results/")

if __name__ == "__main__":
    main()