#!/usr/bin/env python3
"""
Create summary visualizations for attention change analysis
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Load results
results_dir = Path('attention_change_analysis/statistics')
with open(results_dir / 'summary.json', 'r') as f:
    summary = json.load(f)

with open(results_dir / 'attention_change_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. JS Divergence Distribution
ax = axes[0, 0]
all_js = []
for case in results:
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            all_js.append(variant['attention_metrics']['js_divergence'])

ax.hist(all_js, bins=15, edgecolor='black', color='steelblue', alpha=0.7)
ax.axvline(np.mean(all_js), color='red', linestyle='--', label=f'Mean: {np.mean(all_js):.3f}')
ax.set_xlabel('JS Divergence')
ax.set_ylabel('Frequency')
ax.set_title('Attention Change Distribution\n(Lower = More Similar)')
ax.legend()

# 2. Correlation Distribution
ax = axes[0, 1]
all_corr = []
for case in results:
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            all_corr.append(variant['attention_metrics']['correlation'])

ax.hist(all_corr, bins=15, edgecolor='black', color='forestgreen', alpha=0.7)
ax.axvline(np.mean(all_corr), color='red', linestyle='--', label=f'Mean: {np.mean(all_corr):.3f}')
ax.set_xlabel('Correlation')
ax.set_ylabel('Frequency')
ax.set_title('Attention Correlation\n(Higher = More Similar)')
ax.legend()

# 3. Divergence by Answer Change Type
ax = axes[0, 2]
change_types = []
divergences = []

for case in results:
    baseline_answer = case['baseline_answer']
    ground_truth = case['ground_truth']
    
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            variant_answer = variant['answer']
            
            # Classify change type
            if baseline_answer == ground_truth and variant_answer != ground_truth:
                change_type = "Correct→Wrong"
            elif baseline_answer != ground_truth and variant_answer == ground_truth:
                change_type = "Wrong→Correct"
            else:
                change_type = "Wrong→Wrong Different"
            
            change_types.append(change_type)
            divergences.append(variant['attention_metrics']['js_divergence'])

if change_types:
    df_changes = pd.DataFrame({'Change Type': change_types, 'JS Divergence': divergences})
    df_changes.boxplot(column='JS Divergence', by='Change Type', ax=ax)
    ax.set_xlabel('Answer Change Type')
    ax.set_ylabel('JS Divergence')
    ax.set_title('Attention Change by Answer Correction')
    plt.sca(ax)
    plt.xticks(rotation=15)

# 4. Metrics Correlation Matrix
ax = axes[1, 0]
metrics_data = []
for case in results:
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            m = variant['attention_metrics']
            metrics_data.append([
                m['js_divergence'],
                m['correlation'],
                m['l2_distance'],
                m['cosine_similarity']
            ])

if metrics_data:
    df_metrics = pd.DataFrame(metrics_data, 
                              columns=['JS Div', 'Correlation', 'L2 Dist', 'Cosine Sim'])
    corr_matrix = df_metrics.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Attention Metrics Correlation')

# 5. Change by Strategy Type
ax = axes[1, 1]
strategy_divergence = {}
for case in results:
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            strategy = variant['strategy'].split('(')[0].strip()  # Simplify strategy name
            if strategy not in strategy_divergence:
                strategy_divergence[strategy] = []
            strategy_divergence[strategy].append(variant['attention_metrics']['js_divergence'])

# Plot top strategies
top_strategies = sorted(strategy_divergence.items(), 
                       key=lambda x: np.mean(x[1]))[:10]

strategies = [s[0][:20] for s in top_strategies]  # Truncate long names
means = [np.mean(s[1]) for s in top_strategies]
stds = [np.std(s[1]) for s in top_strategies]

y_pos = np.arange(len(strategies))
ax.barh(y_pos, means, xerr=stds, color='coral', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(strategies, fontsize=9)
ax.set_xlabel('Mean JS Divergence')
ax.set_title('Attention Change by Strategy')

# 6. Summary Statistics Table
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""Attention Change Analysis Summary
{'='*35}

Total Cases Analyzed: {summary['num_cases_analyzed']}
Total Comparisons: {summary['total_comparisons']}

JS Divergence:
  Mean: {summary['js_divergence']['mean']:.3f} ± {summary['js_divergence']['std']:.3f}
  Range: [{summary['js_divergence']['min']:.3f}, {summary['js_divergence']['max']:.3f}]

Correlation:
  Mean: {summary['correlation']['mean']:.3f} ± {summary['correlation']['std']:.3f}
  Range: [{summary['correlation']['min']:.3f}, {summary['correlation']['max']:.3f}]

Cosine Similarity:
  Mean: {summary['cosine_similarity']['mean']:.3f} ± {summary['cosine_similarity']['std']:.3f}

Key Finding:
Despite answer changes, attention patterns
remain highly correlated (r={summary['correlation']['mean']:.2f}),
suggesting subtle but impactful shifts
in focus regions."""

ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

plt.suptitle('Attention Pattern Changes When Question Phrasing Alters Answers', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('attention_change_analysis/attention_change_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("KEY INSIGHTS FROM ATTENTION ANALYSIS")
print("="*60)

print(f"\n1. ATTENTION SIMILARITY DESPITE ANSWER CHANGES:")
print(f"   • Mean JS Divergence: {summary['js_divergence']['mean']:.3f} (low divergence)")
print(f"   • Mean Correlation: {summary['correlation']['mean']:.3f} (high similarity)")
print(f"   • Mean Cosine Similarity: {summary['cosine_similarity']['mean']:.3f}")

print(f"\n2. ATTENTION CHANGE PATTERNS:")
# Analyze patterns
correct_to_wrong = []
wrong_to_correct = []

for case in results:
    baseline_answer = case['baseline_answer']
    ground_truth = case['ground_truth']
    
    for variant in case.get('variants', []):
        if variant.get('attention_metrics'):
            variant_answer = variant['answer']
            js_div = variant['attention_metrics']['js_divergence']
            
            if baseline_answer == ground_truth and variant_answer != ground_truth:
                correct_to_wrong.append(js_div)
            elif baseline_answer != ground_truth and variant_answer == ground_truth:
                wrong_to_correct.append(js_div)

if correct_to_wrong:
    print(f"   • Correct→Wrong changes: Mean JS = {np.mean(correct_to_wrong):.3f}")
if wrong_to_correct:
    print(f"   • Wrong→Correct changes: Mean JS = {np.mean(wrong_to_correct):.3f}")

print(f"\n3. MOST SENSITIVE QUESTIONS:")
case_sensitivities = []
for case in results:
    divs = [v['attention_metrics']['js_divergence'] 
            for v in case.get('variants', []) 
            if v.get('attention_metrics')]
    if divs:
        case_sensitivities.append((case['baseline_question'], np.mean(divs)))

case_sensitivities.sort(key=lambda x: x[1], reverse=True)
for q, div in case_sensitivities[:3]:
    print(f"   • '{q[:50]}...' (JS={div:.3f})")

print(f"\n4. IMPLICATIONS:")
print(f"   • Small attention shifts can flip answers")
print(f"   • Model maintains similar overall focus patterns")
print(f"   • Phrasing affects subtle regional emphasis")
print(f"   • High correlation ({summary['correlation']['mean']:.2f}) suggests minor redistributions")

print(f"\nVisualization saved to: attention_change_analysis/attention_change_summary.png")
print(f"Individual comparisons in: attention_change_analysis/comparisons/")
print("="*60)