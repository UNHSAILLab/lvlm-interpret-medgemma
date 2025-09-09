#!/usr/bin/env python3
"""
Fixed evaluation script with proper error handling
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import pandas as pd
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from medgemma4b_runner import MedGemma4BEvaluator, load_medgemma4b
from medgemma_attention_extraction import SystemConfig


def main():
    """Run evaluation with fixes"""
    
    # Configuration
    config = SystemConfig(
        model_id="google/medgemma-4b-it",
        device="cuda",
        io_paths={
            "data_csv": Path("/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv"),
            "image_root": Path("/home/bsada1/mimic_cxr_hundred_vqa"),
            "out_dir": Path("./medgemma4b_results_fixed")
        },
        eval={
            "target_terms": [
                "effusion", "pneumothorax", "consolidation", "pneumonia",
                "edema", "cardiomegaly", "atelectasis", "fracture",
                "opacity", "infiltrate", "pleural", "fluid"
            ],
            "hard_positive_analysis": {"enabled": True},
            "robustness": {
                "run_perturbations": False  # Skip for now
            }
        },
        decoding={
            "max_new_tokens": 50,
            "temperature": 0.0,
            "do_sample": False
        }
    )
    
    print("="*70)
    print("MedGemma-4b Evaluation (Fixed)")
    print("="*70)
    print(f"Output: {config.io_paths['out_dir']}")
    print(f"Target terms: {len(config.eval['target_terms'])} terms")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, processor = load_medgemma4b("cuda")
    print("✓ Model loaded")
    
    # Create evaluator
    print("Initializing evaluator...")
    evaluator = MedGemma4BEvaluator(model, processor, config)
    
    # Load dataset
    df = pd.read_csv(config.io_paths["data_csv"])
    print(f"\nDataset: {len(df)} total samples")
    
    # Process samples with better error handling
    sample_limit = 20  # Start with 20 samples
    print(f"Processing {sample_limit} samples...")
    
    all_results = []
    from tqdm import tqdm
    
    for idx, row in tqdm(df.head(sample_limit).iterrows(), total=sample_limit, desc="Processing"):
        try:
            result = evaluator.process_single_sample(
                row,
                config.io_paths["image_root"],
                config.eval["target_terms"]
            )
            
            if result:
                all_results.append(result)
                
                # Save outputs with error handling
                try:
                    evaluator.save_sample_outputs(result)
                except Exception as e:
                    print(f"Warning: Could not save outputs for {row.study_id}: {e}")
                
        except Exception as e:
            print(f"Error processing {row.study_id}: {e}")
            continue
    
    print(f"\n✓ Processed {len(all_results)} samples successfully")
    
    if all_results:
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save CSV
        csv_path = config.io_paths["out_dir"] / "evaluation_results.csv"
        config.io_paths["out_dir"].mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total processed: {len(results_df)}")
        print(f"Accuracy: {results_df['correct'].mean():.3f}")
        
        # Attention modes
        print("\nAttention modes:")
        for mode, count in results_df['attention_mode'].value_counts().items():
            print(f"  {mode}: {count}")
        
        # Average metrics
        print("\nAverage metrics:")
        metric_cols = ['entropy', 'focus', 'inside_body_ratio']
        for metric in metric_cols:
            if metric in results_df.columns:
                print(f"  {metric}: {results_df[metric].mean():.3f}")
        
        # Hard positive analysis
        if "strategy" in results_df.columns:
            print("\nBy strategy:")
            for strategy in results_df['strategy'].unique():
                strat_df = results_df[results_df['strategy'] == strategy]
                print(f"  {strategy}: accuracy={strat_df['correct'].mean():.3f} (n={len(strat_df)})")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()