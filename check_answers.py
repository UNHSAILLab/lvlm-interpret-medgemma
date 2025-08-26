#!/usr/bin/env python3
"""
Quick script to check answer extraction from existing results
"""

import pandas as pd
import json
from pathlib import Path

# Load existing results if available
results_path = Path('batch_analysis_results/statistics/results.csv')
if results_path.exists():
    df = pd.read_csv(results_path)
    
    print(f"Loaded {len(df)} results")
    print("\n=== ANSWER ACCURACY CHECK ===")
    
    # Check if we have the old columns
    if 'generated_answer' in df.columns:
        print("\nOld extraction method:")
        print(f"Correct: {df['is_correct'].sum()}/{len(df)} = {df['is_correct'].mean():.1%}")
        
        # Show some examples
        print("\nSample answers (first 10):")
        for i, row in df.head(10).iterrows():
            print(f"Q: {row['question'][:50]}...")
            print(f"  Ground truth: {row['ground_truth']}")
            print(f"  Generated: {row.get('generated_answer', 'N/A')}")
            print(f"  Text: {row.get('generated_text', 'N/A')[:100]}")
            print(f"  Correct: {row['is_correct']}")
            print()
    
    # Check by ground truth
    if 'ground_truth' in df.columns:
        print("\nBy ground truth:")
        for gt in df['ground_truth'].unique():
            subset = df[df['ground_truth'] == gt]
            print(f"  {gt}: {subset['is_correct'].sum()}/{len(subset)} = {subset['is_correct'].mean():.1%}")
    
    # Check for patterns in wrong answers
    wrong = df[~df['is_correct']]
    if len(wrong) > 0:
        print(f"\nWrong answers: {len(wrong)}")
        print("First 5 wrong answers:")
        for i, row in wrong.head(5).iterrows():
            print(f"Q: {row['question'][:50]}...")
            print(f"  Ground truth: {row['ground_truth']}")  
            print(f"  Generated: {row.get('generated_answer', 'N/A')}")
            print(f"  Text: {row.get('generated_text', 'N/A')[:100]}")
            print()

# Check intermediate results
pkl_path = Path('batch_analysis_results/raw_data/intermediate_results.pkl')
if pkl_path.exists():
    import pickle
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\n=== INTERMEDIATE RESULTS ({len(results)} samples) ===")
    
    # Check if new fields exist
    if results and 'full_generated_text' in results[0]:
        print("\nNew extraction method detected!")
        correct = sum(1 for r in results if r.get('is_correct', False))
        print(f"Correct: {correct}/{len(results)} = {correct/len(results):.1%}")
        
        # Check answer_in_text
        answer_in_text = sum(1 for r in results if r.get('answer_in_text', False))
        print(f"Answer in text: {answer_in_text}/{len(results)} = {answer_in_text/len(results):.1%}")
        
        # Show examples with full text
        print("\nExamples with full generated text:")
        for i, r in enumerate(results[:3]):
            print(f"\nSample {i}:")
            print(f"Q: {r['question'][:50]}...")
            print(f"Ground truth: {r['ground_truth']}")
            print(f"Full text: {r.get('full_generated_text', 'N/A')}")
            print(f"Extracted: {r.get('extracted_answer', 'N/A')}")
            print(f"Correct: {r.get('is_correct', 'N/A')}")
            print(f"Answer in text: {r.get('answer_in_text', 'N/A')}")