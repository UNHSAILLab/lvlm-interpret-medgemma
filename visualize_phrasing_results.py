#!/usr/bin/env python3
"""
Visualize question phrasing sensitivity results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load results
results_dir = Path('question_phrasing_results')
df = pd.read_csv(results_dir / 'phrasing_analysis.csv')
with open(results_dir / 'summary_statistics.json', 'r') as f:
    summary = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Accuracy by strategy
ax = axes[0, 0]
strategies = list(summary['accuracy_by_strategy'].keys())
accuracies = list(summary['accuracy_by_strategy'].values())
bars = ax.bar(strategies, accuracies, color=['#2E86AB', '#A23B72'])
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Question Phrasing Strategy')
ax.set_ylim([0, 1])
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.1%}', ha='center', va='bottom')

# 2. Consistency distribution
ax = axes[0, 1]
consistency_by_group = df.groupby('baseline_question')['group_consistent'].first()
ax.hist(df.groupby('baseline_question')['consistency_rate'].first(), 
        bins=10, color='#3A8E4C', edgecolor='black')
ax.set_xlabel('Consistency Rate')
ax.set_ylabel('Number of Question Groups')
ax.set_title(f'Consistency Distribution\n(Mean: {summary["overall_consistency"]["mean"]:.1%})')
ax.axvline(summary['overall_consistency']['mean'], color='red', linestyle='--', 
          label=f'Mean: {summary["overall_consistency"]["mean"]:.1%}')
ax.legend()

# 3. Correct vs Incorrect by variant
ax = axes[0, 2]
variant_accuracy = df.groupby('variant_index')['is_correct'].mean()
ax.plot(variant_accuracy.index, variant_accuracy.values, marker='o', linewidth=2)
ax.set_xlabel('Variant Index')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Variant Index')
ax.set_xticks(variant_accuracy.index)
ax.grid(True, alpha=0.3)

# 4. Strategy comparison for each variant
ax = axes[1, 0]
pivot_data = df.pivot_table(values='is_correct', index='variant_index', 
                            columns='strategy', aggfunc='mean')
pivot_data.plot(kind='bar', ax=ax)
ax.set_xlabel('Variant Index')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison Across Variants')
ax.legend(title='Strategy')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 5. Consistency vs Ground Truth
ax = axes[1, 1]
truth_consistency = df.groupby(['baseline_question', 'ground_truth']).agg({
    'consistency_rate': 'first'
}).reset_index()
for truth in ['yes', 'no']:
    data = truth_consistency[truth_consistency['ground_truth'] == truth]['consistency_rate']
    ax.violinplot([data], positions=[0 if truth == 'no' else 1], widths=0.4,
                  showmeans=True, showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(['No', 'Yes'])
ax.set_xlabel('Ground Truth Answer')
ax.set_ylabel('Consistency Rate')
ax.set_title('Response Consistency by Ground Truth')

# 6. Changed answers analysis
ax = axes[1, 2]
changed_groups = summary['groups_with_changed_answers']
if changed_groups:
    # Count reasons for changes
    change_patterns = []
    for group in changed_groups:
        gt = group['ground_truth']
        baseline = group['answers']['baseline']
        if baseline != gt:
            change_patterns.append('Baseline Wrong')
        else:
            change_patterns.append('Baseline Right, Variant Wrong')
    
    from collections import Counter
    pattern_counts = Counter(change_patterns)
    ax.bar(pattern_counts.keys(), pattern_counts.values(), color=['#D62828', '#F77F00'])
    ax.set_ylabel('Number of Groups')
    ax.set_title('Pattern of Inconsistent Responses')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

plt.suptitle('MedGemma Question Phrasing Sensitivity Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(results_dir / 'phrasing_analysis_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)
print(f"✓ Overall consistency: {summary['overall_consistency']['mean']:.1%}")
print(f"✓ Baseline accuracy: {summary['accuracy_by_strategy']['baseline']:.1%}")
print(f"✓ Fallback phrasing accuracy: {summary['accuracy_by_strategy']['fallback phrasing']:.1%}")
print(f"✓ Fully consistent groups: {summary['overall_consistency']['fully_consistent_groups']}/20")
print(f"✓ Groups with changed answers: {summary['num_groups_changed']}/20")

# Additional analysis
print("\n" + "="*60)
print("DETAILED VARIANT ANALYSIS:")
print("="*60)

# Check which specific phrasings cause issues
variant_performance = df.groupby(['variant_index', 'strategy']).agg({
    'is_correct': 'mean',
    'question': 'first'
}).reset_index()

# Find worst performing variants
worst_variants = variant_performance.nsmallest(5, 'is_correct')
print("\nWorst performing phrasings:")
for _, row in worst_variants.iterrows():
    print(f"  Variant {row['variant_index']}: {row['is_correct']:.1%} - Example: '{row['question'][:50]}...'")

# Check if certain medical terms are more sensitive
df['medical_term'] = df['baseline_question'].str.extract(r'(atelectasis|effusion|cardiomegaly|consolidation|edema|pneumonia|opacity)', expand=False)
term_consistency = df[df['medical_term'].notna()].groupby('medical_term')['consistency_rate'].mean()

print("\nConsistency by medical term:")
for term, consistency in term_consistency.items():
    print(f"  {term}: {consistency:.1%}")

print(f"\nVisualization saved to: {results_dir / 'phrasing_analysis_visualization.png'}")