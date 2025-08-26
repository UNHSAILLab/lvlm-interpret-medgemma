#!/usr/bin/env python3
"""
Fix existing results with correct accuracy calculation
"""

import pandas as pd
import json
from pathlib import Path

# Load existing results
df = pd.read_csv('batch_analysis_results/statistics/results.csv')

print(f"Loaded {len(df)} results")

# Fix the accuracy calculation
df['is_correct'] = df['ground_truth'].str.lower() == df['generated_answer'].str.lower()

# Save corrected results
df.to_csv('batch_analysis_results/statistics/results_fixed.csv', index=False)

# Recalculate statistics
stats = {}

# 1. Answer Accuracy Statistics
stats['accuracy'] = {
    'overall': df['is_correct'].mean(),
    'by_answer': df.groupby('ground_truth')['is_correct'].mean().to_dict(),
    'by_term': df.groupby('medical_term')['is_correct'].mean().to_dict() if 'medical_term' in df.columns else {}
}

# 2. Attention Metrics Statistics
attention_metrics = ['inside_body_ratio', 'border_fraction', 'attention_entropy']
for metric in attention_metrics:
    if metric in df.columns:
        stats[metric] = {
            'mean': df[metric].mean(),
            'std': df[metric].std(),
            'median': df[metric].median(),
            'q25': df[metric].quantile(0.25),
            'q75': df[metric].quantile(0.75)
        }

# 3. Regional Distribution
regional_metrics = ['left_fraction', 'right_fraction', 'apical_fraction', 'basal_fraction']
regional_stats = {}
for metric in regional_metrics:
    if metric in df.columns:
        regional_stats[metric] = df[metric].mean()
stats['regional_distribution'] = regional_stats

# 4. Method Success Rates
method_cols = [col for col in df.columns if col.endswith('_success')]
if method_cols:
    stats['method_success_rates'] = {
        col: df[col].mean() for col in method_cols
    }

# Save corrected statistics
with open('batch_analysis_results/statistics/summary_statistics_fixed.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)

# Print summary
print("\n=== CORRECTED RESULTS ===")
print(f"Overall Accuracy: {stats['accuracy']['overall']:.1%}")
print(f"By Ground Truth:")
for truth, acc in stats['accuracy']['by_answer'].items():
    print(f"  {truth}: {acc:.1%}")

if 'medical_term' in df.columns:
    print(f"\nBy Medical Term:")
    for term, acc in stats['accuracy']['by_term'].items():
        if pd.notna(term):
            print(f"  {term}: {acc:.1%}")

print("\nAttention Metrics:")
for metric in attention_metrics:
    if metric in stats:
        print(f"  {metric}: {stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}")

print("\nResults saved to:")
print("  - batch_analysis_results/statistics/results_fixed.csv")
print("  - batch_analysis_results/statistics/summary_statistics_fixed.json")