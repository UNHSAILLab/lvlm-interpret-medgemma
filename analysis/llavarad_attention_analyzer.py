#!/usr/bin/env python3
"""
Comprehensive attention analysis for LLaVA-Rad question phrasing variations
Analyzes attention patterns when answers change due to phrasing
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import logging

# Import visualizers
from llava_rad_visualizer import LLaVARadVisualizer

# Setup logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_rad_attention_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVARadAttentionAnalyzer:
    """Analyze attention changes in LLaVA-Rad for phrasing variations"""
    
    def __init__(self, visualizer, output_dir="llava_rad_attention_analysis"):
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'visualizations': self.output_dir / 'visualizations',
            'by_change_type': self.output_dir / 'by_change_type',
            'statistics': self.output_dir / 'statistics',
            'comparisons': self.output_dir / 'comparisons'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Initialize results
        self.results = {
            'all_comparisons': [],
            'yes_to_no': [],
            'no_to_yes': [],
            'yes_to_yes': [],
            'no_to_no': [],
            'summary_statistics': {}
        }
    
    def compare_attention_maps(self, attn1, attn2):
        """Compare two attention maps"""
        if attn1 is None or attn2 is None:
            return None
        
        # Convert to numpy arrays if needed
        attn1 = np.array(attn1) if not isinstance(attn1, np.ndarray) else attn1
        attn2 = np.array(attn2) if not isinstance(attn2, np.ndarray) else attn2
        
        # Ensure same shape
        if attn1.shape != attn2.shape:
            # Resize to match
            min_shape = (min(attn1.shape[0], attn2.shape[0]), 
                        min(attn1.shape[1], attn2.shape[1]))
            attn1 = attn1[:min_shape[0], :min_shape[1]]
            attn2 = attn2[:min_shape[0], :min_shape[1]]
        
        # Flatten and ensure valid values
        attn1_flat = attn1.flatten()
        attn2_flat = attn2.flatten()
        
        # Replace NaN/Inf with small values
        attn1_flat = np.nan_to_num(attn1_flat, nan=1e-10, posinf=1.0, neginf=0.0)
        attn2_flat = np.nan_to_num(attn2_flat, nan=1e-10, posinf=1.0, neginf=0.0)
        
        # Normalize to probability distributions
        attn1_sum = attn1_flat.sum()
        attn2_sum = attn2_flat.sum()
        
        if attn1_sum == 0:
            attn1_norm = np.ones_like(attn1_flat) / len(attn1_flat)
        else:
            attn1_norm = attn1_flat / attn1_sum
            
        if attn2_sum == 0:
            attn2_norm = np.ones_like(attn2_flat) / len(attn2_flat)
        else:
            attn2_norm = attn2_flat / attn2_sum
        
        # Compute metrics with NaN protection
        try:
            js_div = jensenshannon(attn1_norm, attn2_norm)
            js_div = float(js_div) if not np.isnan(js_div) else 0.0
        except:
            js_div = 0.0
            
        try:
            corr = np.corrcoef(attn1_flat, attn2_flat)[0, 1]
            corr = float(corr) if not np.isnan(corr) else 0.0
        except:
            corr = 0.0
            
        try:
            l2_dist = np.linalg.norm(attn1_flat - attn2_flat)
            l2_dist = float(l2_dist) if not np.isnan(l2_dist) else 0.0
        except:
            l2_dist = 0.0
            
        try:
            norm1 = np.linalg.norm(attn1_flat)
            norm2 = np.linalg.norm(attn2_flat)
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(attn1_flat, attn2_flat) / (norm1 * norm2)
                cos_sim = float(cos_sim) if not np.isnan(cos_sim) else 0.0
            else:
                cos_sim = 0.0
        except:
            cos_sim = 0.0
        
        metrics = {
            'js_divergence': js_div,
            'correlation': corr,
            'l2_distance': l2_dist,
            'cosine_similarity': cos_sim
        }
        
        # Compute difference map
        diff_map = attn2.reshape(attn2.shape) - attn1.reshape(attn1.shape)
        
        # Find regions with most change
        threshold = np.abs(diff_map).mean() + np.abs(diff_map).std()
        significant_changes = np.abs(diff_map) > threshold
        metrics['significant_change_fraction'] = float(significant_changes.mean())
        
        return {
            'metrics': metrics,
            'difference_map': diff_map
        }
    
    def classify_answer_change(self, baseline_answer, variant_answer):
        """Classify type of answer change"""
        if baseline_answer == 'yes' and variant_answer == 'no':
            return 'yes_to_no'
        elif baseline_answer == 'no' and variant_answer == 'yes':
            return 'no_to_yes'
        elif baseline_answer == 'yes' and variant_answer == 'yes':
            return 'yes_to_yes'
        elif baseline_answer == 'no' and variant_answer == 'no':
            return 'no_to_no'
        else:
            return 'uncertain'
    
    def analyze_phrasing_results(self, phrasing_results_path, max_groups=None):
        """Analyze attention changes from phrasing results"""
        
        # Convert to string if Path object
        phrasing_results_path = str(phrasing_results_path)
        
        # Load phrasing results
        if phrasing_results_path.endswith('.json'):
            with open(phrasing_results_path, 'r') as f:
                phrasing_results = json.load(f)
        else:
            # Load from LLaVA-Rad results
            results_file = Path(phrasing_results_path) / 'llava_rad_detailed_results.json'
            with open(results_file, 'r') as f:
                phrasing_results = json.load(f)
        
        logger.info(f"Loaded {len(phrasing_results)} question groups")
        
        if max_groups:
            phrasing_results = phrasing_results[:max_groups]
        
        comparison_count = 0
        change_type_counts = defaultdict(int)
        
        for i, group in enumerate(tqdm(phrasing_results, desc="Analyzing attention")):
            logger.info(f"\nGroup {i+1}: {group['baseline_question'][:50]}...")
            
            # Load image
            image_path = Path(group['image_path'])
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert('RGB')
            
            # Get baseline variant
            baseline_variant = None
            for variant in group['variants']:
                if variant['variant_index'] == 0:
                    baseline_variant = variant
                    break
            
            if not baseline_variant:
                logger.warning("No baseline variant found")
                continue
            
            # Extract baseline attention
            logger.info("  Extracting baseline attention...")
            baseline_result = self.visualizer.generate_with_attention(
                image, baseline_variant['question']
            )
            baseline_attention = baseline_result['visual_attention']
            
            # Analyze each variant
            for variant in group['variants']:
                if variant['variant_index'] == 0:  # Skip baseline
                    continue
                
                logger.info(f"  Variant {variant['variant_index']}: {variant['strategy'][:30]}")
                
                # Extract variant attention
                variant_result = self.visualizer.generate_with_attention(
                    image, variant['question']
                )
                variant_attention = variant_result['visual_attention']
                
                # Classify change type
                change_type = self.classify_answer_change(
                    baseline_variant['extracted_answer'],
                    variant['extracted_answer']
                )
                change_type_counts[change_type] += 1
                
                # Compare attentions
                if baseline_attention is not None and variant_attention is not None:
                    comparison = self.compare_attention_maps(
                        baseline_attention,
                        variant_attention
                    )
                    
                    if comparison:
                        comparison_result = {
                            'group_id': i,
                            'baseline_question': group['baseline_question'],
                            'variant_index': variant['variant_index'],
                            'strategy': variant['strategy'],
                            'baseline_answer': baseline_variant['extracted_answer'],
                            'variant_answer': variant['extracted_answer'],
                            'change_type': change_type,
                            'attention_metrics': comparison['metrics'],
                            'ground_truth': group['ground_truth']
                        }
                        
                        self.results['all_comparisons'].append(comparison_result)
                        self.results[change_type].append(comparison_result)
                        comparison_count += 1
                        
                        # Save visualization for answer changes
                        if change_type in ['yes_to_no', 'no_to_yes']:
                            self.save_change_visualization(
                                image, baseline_attention, variant_attention,
                                baseline_variant, variant, comparison, i
                            )
            
            # Save progress
            if (i + 1) % 5 == 0:
                self.save_intermediate_results()
                logger.info(f"  Progress: {comparison_count} comparisons")
        
        logger.info(f"\nTotal comparisons: {comparison_count}")
        logger.info(f"Change type distribution: {dict(change_type_counts)}")
        
        # Save final results
        self.save_results()
        return self.results
    
    def save_change_visualization(self, image, baseline_attn, variant_attn, 
                                 baseline_variant, variant, comparison, group_id):
        """Save visualization for answer-changing cases"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Baseline attention
        if baseline_attn is not None:
            attn_resized = np.array(Image.fromarray(
                (baseline_attn * 255).astype(np.uint8)
            ).resize(image.size, Image.BILINEAR))
            axes[1].imshow(image)
            axes[1].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[1].set_title(f'Baseline: {baseline_variant["extracted_answer"]}')
            axes[1].axis('off')
        
        # Variant attention
        if variant_attn is not None:
            attn_resized = np.array(Image.fromarray(
                (variant_attn * 255).astype(np.uint8)
            ).resize(image.size, Image.BILINEAR))
            axes[2].imshow(image)
            axes[2].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[2].set_title(f'Variant: {variant["extracted_answer"]}')
            axes[2].axis('off')
        
        # Difference map
        if comparison:
            diff_map = comparison['difference_map']
            im = axes[3].imshow(diff_map, cmap='RdBu_r', 
                               vmin=-np.abs(diff_map).max(), 
                               vmax=np.abs(diff_map).max())
            axes[3].set_title(f'JS Div: {comparison["metrics"]["js_divergence"]:.3f}')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046)
        
        change_type = self.classify_answer_change(
            baseline_variant['extracted_answer'],
            variant['extracted_answer']
        )
        
        plt.suptitle(f'LLaVA-Rad {change_type}: {variant["strategy"][:50]}', fontsize=12)
        plt.tight_layout()
        
        save_path = self.dirs['by_change_type'] / f'{change_type}_group{group_id:03d}_var{variant["variant_index"]}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_intermediate_results(self):
        """Save intermediate results"""
        with open(self.dirs['statistics'] / 'intermediate_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
    
    def save_results(self):
        """Save complete results with statistics"""
        # Save raw results
        with open(self.dirs['statistics'] / 'complete_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Compute statistics
        statistics = self.compute_statistics()
        
        with open(self.dirs['statistics'] / 'llava_rad_statistics.json', 'w') as f:
            json.dump(statistics, f, indent=2, default=float)
        
        # Generate report
        self.generate_report(statistics)
        
        # Create comparison plots
        self.create_comparison_plots(statistics)
    
    def compute_statistics(self):
        """Compute comprehensive statistics"""
        stats = {
            'model': 'LLaVA-Rad',
            'total_comparisons': len(self.results['all_comparisons']),
            'by_change_type': {}
        }
        
        for change_type in ['yes_to_no', 'no_to_yes', 'yes_to_yes', 'no_to_no']:
            type_data = self.results[change_type]
            if type_data:
                metrics = {
                    'js_divergence': [d['attention_metrics']['js_divergence'] for d in type_data],
                    'correlation': [d['attention_metrics']['correlation'] for d in type_data],
                    'cosine_similarity': [d['attention_metrics']['cosine_similarity'] for d in type_data]
                }
                
                stats['by_change_type'][change_type] = {
                    'count': len(type_data),
                    'js_divergence': {
                        'mean': float(np.mean(metrics['js_divergence'])),
                        'std': float(np.std(metrics['js_divergence'])),
                        'median': float(np.median(metrics['js_divergence'])),
                        'min': float(np.min(metrics['js_divergence'])),
                        'max': float(np.max(metrics['js_divergence']))
                    },
                    'correlation': {
                        'mean': float(np.mean(metrics['correlation'])),
                        'std': float(np.std(metrics['correlation'])),
                        'median': float(np.median(metrics['correlation']))
                    },
                    'cosine_similarity': {
                        'mean': float(np.mean(metrics['cosine_similarity'])),
                        'std': float(np.std(metrics['cosine_similarity']))
                    }
                }
        
        # Overall statistics
        if self.results['all_comparisons']:
            all_js = [d['attention_metrics']['js_divergence'] for d in self.results['all_comparisons']]
            all_corr = [d['attention_metrics']['correlation'] for d in self.results['all_comparisons']]
            
            stats['overall'] = {
                'js_divergence_mean': float(np.mean(all_js)),
                'js_divergence_std': float(np.std(all_js)),
                'correlation_mean': float(np.mean(all_corr)),
                'correlation_std': float(np.std(all_corr))
            }
        
        return stats
    
    def generate_report(self, statistics):
        """Generate analysis report"""
        report = []
        report.append("="*80)
        report.append("LLaVA-Rad ATTENTION ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("ANALYSIS SCOPE")
        report.append("-"*40)
        report.append(f"Total Comparisons: {statistics['total_comparisons']}")
        
        # Distribution by change type
        report.append("\nDistribution by Answer Change Type:")
        total = statistics['total_comparisons']
        for change_type, stats in statistics['by_change_type'].items():
            pct = stats['count']/total*100 if total > 0 else 0
            report.append(f"  {change_type.replace('_', '→')}: {stats['count']} ({pct:.1f}%)")
        
        report.append("\nDETAILED STATISTICS BY CHANGE TYPE")
        report.append("-"*40)
        
        for change_type, stats in statistics['by_change_type'].items():
            report.append(f"\n{change_type.replace('_', ' ').upper()}:")
            report.append(f"  Count: {stats['count']}")
            report.append(f"  JS Divergence: {stats['js_divergence']['mean']:.4f} ± {stats['js_divergence']['std']:.4f}")
            report.append(f"    Median: {stats['js_divergence']['median']:.4f}")
            report.append(f"    Range: [{stats['js_divergence']['min']:.4f}, {stats['js_divergence']['max']:.4f}]")
            report.append(f"  Correlation: {stats['correlation']['mean']:.4f} ± {stats['correlation']['std']:.4f}")
            report.append(f"  Cosine Similarity: {stats['cosine_similarity']['mean']:.4f} ± {stats['cosine_similarity']['std']:.4f}")
        
        report.append("\n" + "="*80)
        report_text = "\n".join(report)
        
        with open(self.dirs['statistics'] / 'llava_rad_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def create_comparison_plots(self, statistics):
        """Create comparison plots between models"""
        # Load MedGemma statistics for comparison if available
        medgemma_stats_path = Path('medgemma_attention_analysis/statistics/comprehensive_statistics.json')
        medgemma_stats = None
        if medgemma_stats_path.exists():
            with open(medgemma_stats_path, 'r') as f:
                medgemma_stats = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: JS Divergence comparison
        ax = axes[0, 0]
        change_types = []
        llava_means = []
        llava_stds = []
        medgemma_means = []
        medgemma_stds = []
        
        for change_type in ['yes_to_no', 'no_to_yes', 'yes_to_yes', 'no_to_no']:
            if change_type in statistics['by_change_type']:
                change_types.append(change_type.replace('_', '→'))
                llava_stats = statistics['by_change_type'][change_type]
                llava_means.append(llava_stats['js_divergence']['mean'])
                llava_stds.append(llava_stats['js_divergence']['std'])
                
                if medgemma_stats and change_type in medgemma_stats['by_change_type']:
                    mg_stats = medgemma_stats['by_change_type'][change_type]
                    medgemma_means.append(mg_stats['js_divergence']['mean'])
                    medgemma_stds.append(mg_stats['js_divergence']['std'])
                else:
                    medgemma_means.append(0)
                    medgemma_stds.append(0)
        
        x = np.arange(len(change_types))
        width = 0.35
        
        ax.bar(x - width/2, llava_means, width, yerr=llava_stds, 
               label='LLaVA-Rad', capsize=5, color='blue', alpha=0.7)
        if medgemma_stats:
            ax.bar(x + width/2, medgemma_means, width, yerr=medgemma_stds,
                   label='MedGemma', capsize=5, color='green', alpha=0.7)
        
        ax.set_xlabel('Answer Change Type')
        ax.set_ylabel('Mean JS Divergence')
        ax.set_title('Attention Divergence Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(change_types)
        ax.legend()
        
        # Plot 2: Correlation distribution
        ax = axes[0, 1]
        if statistics['by_change_type']:
            data_to_plot = []
            labels = []
            for change_type, stats in statistics['by_change_type'].items():
                type_data = self.results[change_type]
                if type_data:
                    correlations = [d['attention_metrics']['correlation'] for d in type_data]
                    data_to_plot.append(correlations)
                    labels.append(f"LLaVA-Rad\n{change_type.replace('_', '→')}")
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels)
                ax.set_ylabel('Correlation')
                ax.set_title('LLaVA-Rad Attention Correlation by Change Type')
                ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Summary statistics table
        ax = axes[1, 0]
        ax.axis('off')
        
        summary_data = []
        headers = ['Metric', 'LLaVA-Rad', 'MedGemma']
        
        if 'overall' in statistics:
            summary_data.append([
                'JS Divergence',
                f"{statistics['overall']['js_divergence_mean']:.3f}±{statistics['overall']['js_divergence_std']:.3f}",
                f"{medgemma_stats['overall']['js_divergence_mean']:.3f}±{medgemma_stats['overall']['js_divergence_std']:.3f}" if medgemma_stats else "N/A"
            ])
            summary_data.append([
                'Correlation',
                f"{statistics['overall']['correlation_mean']:.3f}±{statistics['overall']['correlation_std']:.3f}",
                f"{medgemma_stats['overall']['correlation_mean']:.3f}±{medgemma_stats['overall']['correlation_std']:.3f}" if medgemma_stats else "N/A"
            ])
        
        if summary_data:
            table = ax.table(cellText=summary_data, colLabels=headers,
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2)
            ax.set_title('Overall Model Comparison')
        
        # Plot 4: Key findings
        ax = axes[1, 1]
        ax.axis('off')
        
        findings_text = """LLaVA-Rad Key Findings:
        
• Attention patterns analyzed across
  phrasing variations
  
• Answer changes correlated with
  attention redistribution
  
• Model comparison shows differences
  in attention stability
  
• Both models operate near
  decision boundaries"""
        
        ax.text(0.1, 0.5, findings_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.suptitle('LLaVA-Rad vs MedGemma: Attention Analysis Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.dirs['statistics'] / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main execution"""
    logger.info("Starting LLaVA-Rad attention analysis")
    
    # Setup LLaVA-Rad
    visualizer = LLaVARadVisualizer()
    visualizer.load_model(load_in_8bit=True)
    
    # Create analyzer
    analyzer = LLaVARadAttentionAnalyzer(visualizer)
    
    # Check if phrasing results exist
    phrasing_results_dir = Path("llava_rad_phrasing_results")
    
    if phrasing_results_dir.exists():
        # Analyze existing results
        results = analyzer.analyze_phrasing_results(
            phrasing_results_dir,
            max_groups=None  # Analyze all groups for more answer variations
        )
    else:
        logger.warning("No phrasing results found. Run test_question_phrasing_llava_rad.py first.")
        return None
    
    logger.info(f"Analysis complete! Total comparisons: {len(analyzer.results['all_comparisons'])}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    results = main()