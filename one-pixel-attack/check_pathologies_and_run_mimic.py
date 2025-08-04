#!/usr/bin/env python
"""Check available pathologies and run MIMIC test with appropriate settings"""

import torch
import torchxrayvision as xrv
import subprocess
import sys

def check_available_pathologies():
    """List all pathologies available in the surrogate model"""
    print("Loading model to check available pathologies...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    print("\nAvailable pathologies in the model:")
    for idx, pathology in enumerate(model.pathologies):
        print(f"{idx}: {pathology}")
    
    return model.pathologies

def run_mimic_test_fast():
    """Run MIMIC test with reduced iterations for speed"""
    
    pathologies = check_available_pathologies()
    
    # Find a suitable pathology - prefer Effusion or Pneumonia
    target_pathology = "Pneumonia"  # default
    
    if "Effusion" in pathologies:
        target_pathology = "Effusion"
        print(f"\nUsing pathology: {target_pathology}")
    elif "Pleural Effusion" in pathologies:
        target_pathology = "Pleural Effusion"
        print(f"\nUsing pathology: {target_pathology}")
    elif "Pneumonia" in pathologies:
        target_pathology = "Pneumonia"
        print(f"\nUsing pathology: {target_pathology}")
    else:
        print(f"\nWarning: Common pathologies not found. Using: {pathologies[0]}")
        target_pathology = pathologies[0]
    
    print(f"\nRunning MIMIC test with reduced iterations for speed...")
    print(f"Target pathology: {target_pathology}")
    print("Max iterations: 20 (reduced from 100)")
    print("Population size: 50 (reduced from 200)")
    
    cmd = [
        sys.executable,
        "run_robustness_evaluation.py",
        "--data-path", "./mimic_adapted_questions.csv",
        "--source-folder", "/home/bsada1/mimic_cxr_hundred_vqa",
        "--output-folder", "/home/bsada1/mimic_cxr_hundred_vqa/onepixel_attack",
        "--preprocessed-folder", "/home/bsada1/mimic_cxr_hundred_vqa/preprocessed_only",
        "--results-folder", "./mimic_robustness_results",
        "--include-preprocessed-eval",
        "--pathology", target_pathology,
        "--max-iter", "20",  # Reduced for speed
        "--pop-size", "50",  # Reduced for speed
        "--limit", "10",     # Test on just 10 samples first
        "--skip-preprocessing"  # Skip since already done
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_mimic_test_fast()