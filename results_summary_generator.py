#!/usr/bin/env python3
"""
Summarize the full phrasing analysis results
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_results():
    results_dir = Path('question_phrasing_results')
    
    # Load results
    with open(results_dir / 'summary_statistics.json', 'r') as f:
        summary = json.load(f)
    
    df = pd.read_csv(results_dir / 'phrasing_analysis.csv')
    
    print("="*80)
    print("FULL MEDGEMMA QUESTION PHRASING SENSITIVITY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Total question groups analyzed: {summary['total_groups']}")
    print(f"  • Total variants tested: {summary['total_variants_tested']}")
    print(f"  • Average variants per question: {summary['total_variants_tested'] / summary['total_groups']:.1f}")
    
    print(f"\n🎯 CONSISTENCY METRICS:")
    print(f"  • Mean consistency: {summary['overall_consistency']['mean']:.1%}")
    print(f"  • Std deviation: {summary['overall_consistency']['std']:.1%}")
    print(f"  • Range: {summary['overall_consistency']['min']:.1%} - {summary['overall_consistency']['max']:.1%}")
    print(f"  • Fully consistent groups: {summary['overall_consistency']['fully_consistent_groups']}/{summary['total_groups']} ({summary['overall_consistency']['fully_consistent_groups']/summary['total_groups']*100:.1f}%)")
    print(f"  • Groups with variations: {summary['num_groups_changed']}/{summary['total_groups']} ({summary['num_groups_changed']/summary['total_groups']*100:.1f}%)")
    
    print(f"\n📈 ACCURACY BY STRATEGY:")
    for strategy, acc in summary['accuracy_by_strategy'].items():
        print(f"  • {strategy}: {acc:.1%}")
    
    if len(summary['accuracy_by_strategy']) > 1:
        accuracies = list(summary['accuracy_by_strategy'].values())
        diff = max(accuracies) - min(accuracies)
        print(f"  • Max difference: {diff:.1%}")
    
    print(f"\n🔍 DETAILED VARIANT ANALYSIS:")
    
    # Analyze by variant type
    variant_types = df['strategy'].value_counts()
    print(f"\n  Variant type distribution:")
    for vtype, count in variant_types.items()[:10]:
        print(f"    • {vtype}: {count} instances")
    
    # Find most problematic phrasings
    variant_accuracy = df.groupby('strategy')['is_correct'].mean().sort_values()
    print(f"\n  Worst performing strategies:")
    for strategy, acc in variant_accuracy.head(5).items():
        print(f"    • {strategy}: {acc:.1%}")
    
    print(f"\n  Best performing strategies:")
    for strategy, acc in variant_accuracy.tail(5).items():
        print(f"    • {strategy}: {acc:.1%}")
    
    # Analyze medical terms if available
    df['medical_term'] = df['baseline_question'].str.extract(
        r'(atelectasis|effusion|cardiomegaly|consolidation|edema|pneumonia|'
        r'opacity|pneumothorax|fracture|hernia|calcification|bronchiectasis|'
        r'infiltrate|nodule|mass)', expand=False
    )
    
    if df['medical_term'].notna().any():
        print(f"\n🏥 MEDICAL CONDITION ANALYSIS:")
        
        # Consistency by medical term
        term_consistency = df[df['medical_term'].notna()].groupby('medical_term')['consistency_rate'].mean()
        print(f"\n  Consistency by medical condition:")
        for term, consistency in term_consistency.sort_values(ascending=False).items():
            count = df[df['medical_term'] == term]['baseline_question'].nunique()
            print(f"    • {term}: {consistency:.1%} (n={count} questions)")
        
        # Accuracy by medical term
        term_accuracy = df[df['medical_term'].notna()].groupby('medical_term')['is_correct'].mean()
        print(f"\n  Accuracy by medical condition:")
        for term, acc in term_accuracy.sort_values(ascending=False).items():
            print(f"    • {term}: {acc:.1%}")
    
    # Find patterns in inconsistent responses
    if summary['groups_with_changed_answers']:
        print(f"\n⚠️ INCONSISTENT RESPONSE PATTERNS:")
        print(f"  Total groups with variations: {len(summary['groups_with_changed_answers'])}")
        
        # Analyze pattern types
        patterns = []
        for group in summary['groups_with_changed_answers']:
            answers = list(set(group['answers'].values()))
            if 'yes' in answers and 'no' in answers:
                patterns.append('yes/no flip')
            elif 'uncertain' in answers:
                patterns.append('uncertain response')
            else:
                patterns.append('other')
        
        from collections import Counter
        pattern_counts = Counter(patterns)
        print(f"\n  Types of inconsistencies:")
        for pattern, count in pattern_counts.items():
            print(f"    • {pattern}: {count} cases")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Determine robustness level
    if summary['overall_consistency']['mean'] >= 0.95:
        robustness = "Excellent"
    elif summary['overall_consistency']['mean'] >= 0.90:
        robustness = "Good"
    elif summary['overall_consistency']['mean'] >= 0.80:
        robustness = "Moderate"
    else:
        robustness = "Poor"
    
    print(f"  • Robustness level: {robustness} ({summary['overall_consistency']['mean']:.1%} consistency)")
    
    # Performance drop analysis
    if 'baseline' in summary['accuracy_by_strategy']:
        baseline_acc = summary['accuracy_by_strategy']['baseline']
        avg_variant_acc = np.mean([acc for k, acc in summary['accuracy_by_strategy'].items() if k != 'baseline'])
        drop = baseline_acc - avg_variant_acc
        
        print(f"  • Baseline accuracy: {baseline_acc:.1%}")
        print(f"  • Average variant accuracy: {avg_variant_acc:.1%}")
        print(f"  • Performance drop with variations: {drop:.1%}")
    
    print("\n" + "="*80)
    print(f"Analysis saved to: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    analyze_results()