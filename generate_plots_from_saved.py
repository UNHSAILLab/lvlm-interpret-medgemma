#!/usr/bin/env python3
"""
Generate plots from saved batch analysis results
"""

import sys
import pandas as pd
import pickle
from pathlib import Path

# Add path for imports
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from batch_analysis_100samples import BatchAnalyzer

# Load saved results
results_path = Path('batch_analysis_results/statistics/results.csv')
df = pd.read_csv(results_path)
results = df.to_dict('records')

print(f"Loaded {len(results)} results")

# Create a mock analyzer instance (we only need it for the plotting methods)
import torch

class MockModel:
    def parameters(self):
        return iter([torch.tensor([0.0])])

class MockDataLoader:
    df = pd.DataFrame()

analyzer = BatchAnalyzer(MockModel(), None, MockDataLoader(), "batch_analysis_results")
analyzer.device = torch.device('cpu')

# Generate statistics
print("Generating statistics...")
stats = analyzer.generate_statistics(results)

# Generate plots
print("Generating plots...")
analyzer.generate_plots(results)

# Generate report  
print("Generating report...")
analyzer.generate_report(results, stats)

print("Done! Check batch_analysis_results/statistics/ for plots and report")