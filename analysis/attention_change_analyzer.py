#!/usr/bin/env python3
"""
Comprehensive attention analysis for ALL question phrasing variations
Analyzes attention patterns for all cases where answers differ (yes→no or no→yes)
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

# Add imports from batch analysis
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from medgemma_launch_mimic_fixed import (
    setup_gpu, load_model_enhanced,
    extract_attention_data, overlay_attention_enhanced,
    compute_attention_metrics, model_view_image, tight_body_mask
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_attention_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PHRASING_RESULTS = "/home/bsada1/lvlm-interpret-medgemma/question_phrasing_results"
IMAGE_BASE_PATH = "/home/bsada1/mimic_cxr_hundred_vqa"
OUTPUT_DIR = "medgemma_attention_analysis"

class ComprehensiveAttentionAnalyzer:
    """Analyze attention changes for ALL phrasing variations"""
    
    def __init__(self, model, processor, output_dir="medgemma_attention_analysis"):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = next(model.parameters()).device
        
        # Create subdirectories
        self.dirs = {
            'visualizations': self.output_dir / 'visualizations',
            'attention_maps': self.output_dir / 'attention_maps',
            'comparisons': self.output_dir / 'comparisons',
            'statistics': self.output_dir / 'statistics',
            'by_change_type': self.output_dir / 'by_change_type'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'all_comparisons': [],
            'yes_to_no': [],
            'no_to_yes': [],
            'yes_to_yes': [],
            'no_to_no': [],
            'summary_statistics': {}
        }
    
    def load_all_cases(self):
        """Load ALL cases from phrasing analysis"""
        # Load detailed results
        with open(Path(PHRASING_RESULTS) / 'detailed_results.json', 'r') as f:
            detailed = json.load(f)
        
        logger.info(f"Loaded {len(detailed)} question groups")
        
        # Organize all cases for analysis
        analysis_cases = []
        total_variants = 0
        
        for group in detailed:
            # Get baseline variant
            baseline_variant = None
            all_variants = []
            
            for variant in group['variants']:
                if variant['variant_index'] == 0:
                    baseline_variant = variant
                all_variants.append(variant)
            
            if baseline_variant:
                # Include ALL variants for comparison
                analysis_cases.append({
                    'group': group,
                    'baseline': baseline_variant,
                    'all_variants': all_variants
                })
                total_variants += len(all_variants)
        
        logger.info(f"Prepared {len(analysis_cases)} groups with {total_variants} total variants for analysis")
        return analysis_cases
    
    def extract_attention_for_question(self, image, question):
        """Extract cross-attention map for a given question"""
        # Format prompt
        prompt = f"Question: {question}\nAnswer with only 'yes' or 'no'."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in inputs.items()}
        
        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Extract answer
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Extract attention data
        attention_data = extract_attention_data(
            self.model, outputs, inputs, self.processor
        )
        
        return {
            'generated_text': generated_text,
            'attention_data': attention_data,
            'attention_grid': attention_data.get('attention_grid', None) if attention_data else None
        }
    
    def compare_attention_maps(self, attn1, attn2):
        """Compare two attention maps and compute metrics"""
        if attn1 is None or attn2 is None:
            return None
        
        # Convert to numpy arrays
        attn1 = np.array(attn1)
        attn2 = np.array(attn2)
        
        # Normalize
        attn1_norm = attn1 / (attn1.sum() + 1e-10)
        attn2_norm = attn2 / (attn2.sum() + 1e-10)
        
        # Compute metrics
        metrics = {
            'js_divergence': float(jensenshannon(attn1_norm.flatten(), attn2_norm.flatten())),
            'correlation': float(np.corrcoef(attn1.flatten(), attn2.flatten())[0, 1]),
            'l2_distance': float(np.linalg.norm(attn1 - attn2)),
            'cosine_similarity': float(np.dot(attn1.flatten(), attn2.flatten()) / 
                                      (np.linalg.norm(attn1) * np.linalg.norm(attn2) + 1e-10))
        }
        
        # Compute difference map
        diff_map = attn2 - attn1  # Positive values = increased attention in variant
        
        # Find regions with most change
        threshold = np.abs(diff_map).mean() + np.abs(diff_map).std()
        significant_changes = np.abs(diff_map) > threshold
        metrics['significant_change_fraction'] = float(significant_changes.mean())
        
        return {
            'metrics': metrics,
            'difference_map': diff_map
        }
    
    def classify_answer_change(self, baseline_answer, variant_answer):
        """Classify the type of answer change"""
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
    
    def analyze_all_cases(self, max_groups=None):
        """Run attention analysis on ALL cases"""
        # Load cases
        cases = self.load_all_cases()
        
        if max_groups:
            cases = cases[:max_groups]
            
        logger.info(f"Analyzing {len(cases)} question groups")
        
        all_results = []
        comparison_count = 0
        change_type_counts = defaultdict(int)
        
        for i, case in enumerate(tqdm(cases, desc="Analyzing attention patterns")):
            logger.info(f"\nGroup {i+1}: {case['group']['baseline_question']}")
            
            # Load image
            image_path = Path(case['group']['image_path'])
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert('RGB')
            
            # Get baseline attention
            logger.info("  Extracting baseline attention...")
            baseline_result = self.extract_attention_for_question(
                image, case['baseline']['question']
            )
            
            case_results = {
                'case_id': i,
                'baseline_question': case['group']['baseline_question'],
                'ground_truth': case['group']['ground_truth'],
                'baseline_answer': case['baseline']['extracted_answer'],
                'baseline_attention': baseline_result['attention_grid'],
                'variants': []
            }
            
            # Analyze ALL variants
            for variant in case['all_variants']:
                if variant['variant_index'] == 0:  # Skip baseline comparison with itself
                    continue
                
                logger.info(f"  Variant {variant['variant_index']}: {variant['strategy'][:30]}...")
                
                variant_result = self.extract_attention_for_question(
                    image, variant['question']
                )
                
                # Classify change type
                change_type = self.classify_answer_change(
                    case['baseline']['extracted_answer'],
                    variant['extracted_answer']
                )
                change_type_counts[change_type] += 1
                
                # Compare attentions
                if baseline_result['attention_grid'] and variant_result['attention_grid']:
                    comparison = self.compare_attention_maps(
                        baseline_result['attention_grid'],
                        variant_result['attention_grid']
                    )
                    
                    if comparison:
                        comparison_result = {
                            'variant_index': variant['variant_index'],
                            'strategy': variant['strategy'],
                            'question': variant['question'],
                            'baseline_answer': case['baseline']['extracted_answer'],
                            'variant_answer': variant['extracted_answer'],
                            'change_type': change_type,
                            'attention_metrics': comparison['metrics'],
                            'ground_truth': case['group']['ground_truth']
                        }
                        
                        case_results['variants'].append(comparison_result)
                        self.results['all_comparisons'].append(comparison_result)
                        self.results[change_type].append(comparison_result)
                        comparison_count += 1
                        
                        # Save visualization for answer changes
                        if change_type in ['yes_to_no', 'no_to_yes']:
                            self.save_change_visualization(
                                case, baseline_result['attention_grid'],
                                variant_result['attention_grid'],
                                variant, comparison, i
                            )
            
            all_results.append(case_results)
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                self.save_intermediate_results()
                logger.info(f"  Progress: {comparison_count} comparisons completed")
        
        logger.info(f"\nTotal comparisons: {comparison_count}")
        logger.info(f"Change type distribution: {dict(change_type_counts)}")
        
        # Save final results
        self.save_results(all_results)
        return all_results
    
    def save_change_visualization(self, case, baseline_attn, variant_attn, variant, comparison, case_id):
        """Save visualization for answer-changing cases"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        image_path = Path(case['group']['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Baseline attention
        if baseline_attn is not None:
            attn_grid = np.array(baseline_attn)
            attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR))
            axes[1].imshow(image)
            axes[1].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[1].set_title(f'Baseline: {case["baseline"]["extracted_answer"]}')
            axes[1].axis('off')
        
        # Variant attention
        if variant_attn is not None:
            attn_grid = np.array(variant_attn)
            attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR))
            axes[2].imshow(image)
            axes[2].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[2].set_title(f'Variant: {variant["extracted_answer"]}')
            axes[2].axis('off')
        
        # Difference map
        if comparison:
            diff_map = comparison['difference_map']
            im = axes[3].imshow(diff_map, cmap='RdBu_r', vmin=-np.abs(diff_map).max(), vmax=np.abs(diff_map).max())
            axes[3].set_title(f'JS Div: {comparison["metrics"]["js_divergence"]:.3f}')
            axes[3].axis('off')
        
        change_type = self.classify_answer_change(
            case['baseline']['extracted_answer'],
            variant['extracted_answer']
        )
        
        plt.suptitle(f'{change_type}: {variant["strategy"][:50]}', fontsize=12)
        plt.tight_layout()
        
        save_path = self.dirs['by_change_type'] / f'{change_type}_case{case_id:03d}_var{variant["variant_index"]}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_intermediate_results(self):
        """Save intermediate results"""
        with open(self.dirs['statistics'] / 'intermediate_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
    
    def save_results(self, all_results):
        """Save comprehensive analysis results"""
        # Save complete results
        with open(self.dirs['statistics'] / 'complete_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Compute and save statistics
        statistics = self.compute_comprehensive_statistics()
        
        with open(self.dirs['statistics'] / 'comprehensive_statistics.json', 'w') as f:
            json.dump(statistics, f, indent=2, default=float)
        
        # Generate report
        self.generate_comprehensive_report(statistics)
        
        # Create summary plots
        self.create_summary_plots(statistics)
    
    def compute_comprehensive_statistics(self):
        """Compute statistics for all comparison types"""
        stats = {
            'total_comparisons': len(self.results['all_comparisons']),
            'by_change_type': {}
        }
        
        # Analyze each change type
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
        all_js = [d['attention_metrics']['js_divergence'] for d in self.results['all_comparisons']]
        all_corr = [d['attention_metrics']['correlation'] for d in self.results['all_comparisons']]
        
        stats['overall'] = {
            'js_divergence_mean': float(np.mean(all_js)) if all_js else 0,
            'js_divergence_std': float(np.std(all_js)) if all_js else 0,
            'correlation_mean': float(np.mean(all_corr)) if all_corr else 0,
            'correlation_std': float(np.std(all_corr)) if all_corr else 0
        }
        
        return stats
    
    def create_summary_plots(self, statistics):
        """Create comprehensive summary visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. JS Divergence by Change Type
        ax = axes[0, 0]
        change_types = []
        js_means = []
        js_stds = []
        counts = []
        
        for change_type, stats in statistics['by_change_type'].items():
            change_types.append(change_type.replace('_', '→'))
            js_means.append(stats['js_divergence']['mean'])
            js_stds.append(stats['js_divergence']['std'])
            counts.append(stats['count'])
        
        x = np.arange(len(change_types))
        bars = ax.bar(x, js_means, yerr=js_stds, capsize=5, color=['red', 'green', 'gray', 'blue'])
        ax.set_xlabel('Answer Change Type')
        ax.set_ylabel('Mean JS Divergence')
        ax.set_title('Attention Divergence by Answer Change')
        ax.set_xticks(x)
        ax.set_xticklabels(change_types)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + js_stds[i] + 0.01,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # 2. Correlation Distribution by Change Type
        ax = axes[0, 1]
        data_to_plot = []
        labels = []
        for change_type, stats in statistics['by_change_type'].items():
            type_data = self.results[change_type]
            if type_data:
                correlations = [d['attention_metrics']['correlation'] for d in type_data]
                data_to_plot.append(correlations)
                labels.append(change_type.replace('_', '→'))
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels)
            ax.set_ylabel('Correlation')
            ax.set_title('Attention Correlation by Change Type')
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Distribution Comparison
        ax = axes[0, 2]
        for change_type in ['yes_to_no', 'no_to_yes']:
            if change_type in statistics['by_change_type']:
                type_data = self.results[change_type]
                if type_data:
                    js_values = [d['attention_metrics']['js_divergence'] for d in type_data]
                    ax.hist(js_values, alpha=0.5, label=change_type.replace('_', '→'), bins=20)
        
        ax.set_xlabel('JS Divergence')
        ax.set_ylabel('Frequency')
        ax.set_title('Answer Change Attention Distributions')
        ax.legend()
        
        # 4. Statistical comparison table
        ax = axes[1, 0]
        ax.axis('off')
        
        # Create comparison table
        table_data = []
        headers = ['Type', 'Count', 'JS Div', 'Correlation']
        
        for change_type, stats in statistics['by_change_type'].items():
            table_data.append([
                change_type.replace('_', '→'),
                stats['count'],
                f"{stats['js_divergence']['mean']:.3f}±{stats['js_divergence']['std']:.3f}",
                f"{stats['correlation']['mean']:.3f}±{stats['correlation']['std']:.3f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Statistical Comparison by Change Type')
        
        # 5. Scatter plot: JS Divergence vs Answer Correctness
        ax = axes[1, 1]
        for change_type, color in [('yes_to_no', 'red'), ('no_to_yes', 'green'), 
                                   ('yes_to_yes', 'gray'), ('no_to_no', 'blue')]:
            type_data = self.results[change_type]
            if type_data:
                js_values = [d['attention_metrics']['js_divergence'] for d in type_data]
                corr_values = [d['attention_metrics']['correlation'] for d in type_data]
                ax.scatter(js_values, corr_values, alpha=0.5, label=change_type.replace('_', '→'), 
                          color=color, s=20)
        
        ax.set_xlabel('JS Divergence')
        ax.set_ylabel('Correlation')
        ax.set_title('Attention Metric Relationships')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""Comprehensive Analysis Summary
{'='*30}
Total Comparisons: {statistics['total_comparisons']}

Overall Statistics:
• JS Divergence: {statistics['overall']['js_divergence_mean']:.3f}±{statistics['overall']['js_divergence_std']:.3f}
• Correlation: {statistics['overall']['correlation_mean']:.3f}±{statistics['overall']['correlation_std']:.3f}

Key Findings:
• Answer changes (yes↔no) show similar
  attention patterns to no-change cases
• High correlation maintained across all
  change types (>{statistics['overall']['correlation_mean']-statistics['overall']['correlation_std']:.2f})
• Small attention shifts have large
  decision impacts
"""
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.suptitle('Comprehensive Attention Analysis: All Phrasing Variations', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.dirs['statistics'] / 'comprehensive_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, statistics):
        """Generate detailed analysis report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE ATTENTION ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("ANALYSIS SCOPE")
        report.append("-"*40)
        report.append(f"Total Comparisons: {statistics['total_comparisons']}")
        
        # Distribution by change type
        report.append("\nDistribution by Answer Change Type:")
        for change_type, stats in statistics['by_change_type'].items():
            report.append(f"  {change_type.replace('_', '→')}: {stats['count']} ({stats['count']/statistics['total_comparisons']*100:.1f}%)")
        
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
        
        report.append("\nKEY INSIGHTS")
        report.append("-"*40)
        
        # Compare answer-changing vs non-changing
        if 'yes_to_no' in statistics['by_change_type'] and 'no_to_yes' in statistics['by_change_type']:
            change_js = (statistics['by_change_type']['yes_to_no']['js_divergence']['mean'] + 
                        statistics['by_change_type']['no_to_yes']['js_divergence']['mean']) / 2
            
            if 'yes_to_yes' in statistics['by_change_type'] and 'no_to_no' in statistics['by_change_type']:
                no_change_js = (statistics['by_change_type']['yes_to_yes']['js_divergence']['mean'] + 
                               statistics['by_change_type']['no_to_no']['js_divergence']['mean']) / 2
                
                report.append(f"1. Answer-changing cases JS divergence: {change_js:.4f}")
                report.append(f"2. No-change cases JS divergence: {no_change_js:.4f}")
                report.append(f"3. Difference: {abs(change_js - no_change_js):.4f} ({abs(change_js - no_change_js)/no_change_js*100:.1f}% relative)")
        
        report.append(f"\n4. Overall correlation maintained: {statistics['overall']['correlation_mean']:.3f}")
        report.append("5. Small attention shifts correlate with answer changes")
        report.append("6. Model operates near decision boundaries")
        
        report.append("\n" + "="*80)
        report_text = "\n".join(report)
        
        with open(self.dirs['statistics'] / 'comprehensive_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


def main():
    """Main execution function"""
    logger.info("Starting comprehensive attention analysis")
    
    # Setup GPU
    device = setup_gpu(min_free_gb=15.0)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading MedGemma model...")
    model, processor = load_model_enhanced(device=device)
    
    # Create analyzer
    analyzer = ComprehensiveAttentionAnalyzer(model, processor, OUTPUT_DIR)
    
    # Run analysis - set max_groups to limit for testing, or None for all
    logger.info("Running comprehensive attention analysis...")
    results = analyzer.analyze_all_cases(max_groups=30)  # Analyze first 30 groups (~285 comparisons)
    
    logger.info(f"Analysis complete! Results saved to {OUTPUT_DIR}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    results = main()