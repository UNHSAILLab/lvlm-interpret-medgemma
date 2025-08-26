#!/usr/bin/env python3
"""
Process the completed batch analysis results
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Add path for imports
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from batch_analysis_100samples import BatchAnalyzer

# Load the completed results
pkl_path = Path('batch_analysis_results/raw_data/intermediate_results.pkl')
if pkl_path.exists():
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} completed results")
    
    # Create a mock analyzer to use its methods
    import torch
    
    class MockModel:
        def parameters(self):
            return iter([torch.tensor([0.0])])
    
    class MockDataLoader:
        df = pd.DataFrame()
    
    analyzer = BatchAnalyzer(MockModel(), None, MockDataLoader(), "batch_analysis_results")
    analyzer.device = torch.device('cpu')
    
    # Save results
    analyzer.results['all_samples'] = results
    analyzer.save_results()
    
    # Generate statistics
    print("Generating statistics...")
    stats = analyzer.generate_statistics(results)
    
    # Generate plots
    print("Generating plots...")
    analyzer.generate_plots(results)
    
    # Generate report
    print("Generating report...")
    analyzer.generate_report(results, stats)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Overall Accuracy: {stats['accuracy']['overall']:.1%}")
    print(f"By Ground Truth:")
    for truth, acc in stats['accuracy']['by_answer'].items():
        print(f"  {truth}: {acc:.1%}")
    
    print("\nProcessing complete!")
    print("Results saved to batch_analysis_results/statistics/")
else:
    print("No intermediate results found!")