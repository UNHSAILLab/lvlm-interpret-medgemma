#!/usr/bin/env python3
"""
Debug accuracy calculation issue
"""

import pandas as pd

# Load existing results
df = pd.read_csv('batch_analysis_results/statistics/results.csv')

print(f"Loaded {len(df)} results")
print(f"\nData types:")
print(f"ground_truth type: {df['ground_truth'].dtype}")
print(f"generated_answer type: {df['generated_answer'].dtype}")

print("\nFirst 5 comparisons:")
for i, row in df.head(5).iterrows():
    gt = row['ground_truth']
    gen = row['generated_answer']
    print(f"Row {i}:")
    print(f"  ground_truth: '{gt}' (type: {type(gt).__name__})")
    print(f"  generated_answer: '{gen}' (type: {type(gen).__name__})")
    print(f"  Are equal?: {gt == gen}")
    print(f"  Lower equal?: {str(gt).lower() == str(gen).lower()}")
    print(f"  Stored is_correct: {row['is_correct']}")
    print()

# Recalculate accuracy with proper comparison
df['correct_calc'] = df['ground_truth'].str.lower() == df['generated_answer'].str.lower()
print(f"\nRecalculated accuracy: {df['correct_calc'].mean():.1%}")

# By ground truth
print("\nBy ground truth (recalculated):")
for gt in df['ground_truth'].unique():
    subset = df[df['ground_truth'] == gt]
    print(f"  {gt}: {subset['correct_calc'].mean():.1%}")