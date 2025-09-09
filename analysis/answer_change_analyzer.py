#!/usr/bin/env python3
"""
Analyze cross-attention patterns for questions where phrasing changes altered answers
Uses the batch analysis framework to extract and compare attention maps
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

# Add imports from batch analysis
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from batch_analysis_100samples import BatchAnalyzer
from medgemma_launch_mimic_fixed import (
    setup_gpu, load_model_enhanced, MIMICDataLoader,
    extract_attention_data, overlay_attention_enhanced,
    compute_attention_metrics, model_view_image, tight_body_mask
)

# Configuration
PHRASING_RESULTS = "/home/bsada1/lvlm-interpret-medgemma/question_phrasing_results"
IMAGE_BASE_PATH = "/home/bsada1/mimic_cxr_hundred_vqa"
OUTPUT_DIR = "attention_change_analysis"

class AttentionChangeAnalyzer:
    """Analyze attention changes when phrasing alters answers"""
    
    def __init__(self, model, processor, output_dir="attention_change_analysis"):
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
            'statistics': self.output_dir / 'statistics'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def load_changed_answer_cases(self):
        """Load cases where phrasing changed the answer"""
        # Load phrasing analysis results
        with open(Path(PHRASING_RESULTS) / 'summary_statistics.json', 'r') as f:
            summary = json.load(f)
        
        # Get groups with changed answers
        changed_groups = summary['groups_with_changed_answers']
        
        # Load detailed results for more info
        with open(Path(PHRASING_RESULTS) / 'detailed_results.json', 'r') as f:
            detailed = json.load(f)
        
        # Filter for cases with actual answer changes (not just consistency < 100%)
        analysis_cases = []
        
        for group in changed_groups[:10]:  # Limit to first 10 for analysis
            baseline_q = group['baseline_question']
            
            # Find this group in detailed results
            for detail_group in detailed:
                if detail_group['baseline_question'] == baseline_q:
                    # Get baseline variant
                    baseline_variant = None
                    changed_variants = []
                    
                    for variant in detail_group['variants']:
                        if variant['variant_index'] == 0:
                            baseline_variant = variant
                        elif variant['extracted_answer'] != baseline_variant['extracted_answer'] if baseline_variant else True:
                            changed_variants.append(variant)
                    
                    if baseline_variant and changed_variants:
                        analysis_cases.append({
                            'group': detail_group,
                            'baseline': baseline_variant,
                            'changed_variants': changed_variants[:2]  # Take up to 2 variants with different answers
                        })
                    break
        
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
        
        return {
            'metrics': metrics,
            'difference_map': diff_map
        }
    
    def visualize_attention_comparison(self, case_data, baseline_attn, variant_attn, variant_info):
        """Create visualization comparing attention patterns"""
        image_path = Path(case_data['group']['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original X-ray')
        axes[0, 0].axis('off')
        
        # Baseline attention
        if baseline_attn is not None:
            attn_grid = np.array(baseline_attn)
            # Resize to image size for overlay
            attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR))
            axes[0, 1].imshow(image)
            axes[0, 1].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[0, 1].set_title(f'Baseline: "{case_data["baseline"]["question"][:50]}..."\nAnswer: {case_data["baseline"]["extracted_answer"]}')
            axes[0, 1].axis('off')
        
        # Variant attention
        if variant_attn is not None:
            attn_grid = np.array(variant_attn)
            attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR))
            axes[0, 2].imshow(image)
            axes[0, 2].imshow(attn_resized, alpha=0.5, cmap='hot')
            axes[0, 2].set_title(f'Variant: "{variant_info["question"][:50]}..."\nAnswer: {variant_info["extracted_answer"]}')
            axes[0, 2].axis('off')
        
        # Attention difference
        if baseline_attn is not None and variant_attn is not None:
            comparison = self.compare_attention_maps(baseline_attn, variant_attn)
            if comparison:
                diff_map = comparison['difference_map']
                
                # Show difference map
                im = axes[1, 0].imshow(diff_map, cmap='RdBu_r', vmin=-np.abs(diff_map).max(), vmax=np.abs(diff_map).max())
                axes[1, 0].set_title('Attention Difference\n(Red=Increased, Blue=Decreased)')
                axes[1, 0].axis('off')
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
                
                # Show metrics
                metrics_text = f"JS Divergence: {comparison['metrics']['js_divergence']:.3f}\n"
                metrics_text += f"Correlation: {comparison['metrics']['correlation']:.3f}\n"
                metrics_text += f"L2 Distance: {comparison['metrics']['l2_distance']:.3f}\n"
                metrics_text += f"Cosine Similarity: {comparison['metrics']['cosine_similarity']:.3f}"
                
                axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                              fontsize=12, verticalalignment='center')
                axes[1, 1].set_title('Attention Metrics')
                axes[1, 1].axis('off')
        
        # Show question details
        details_text = f"Baseline Question:\n{case_data['baseline']['question']}\n\n"
        details_text += f"Variant ({variant_info['strategy']}):\n{variant_info['question']}\n\n"
        details_text += f"Ground Truth: {case_data['group']['ground_truth']}\n"
        details_text += f"Baseline Answer: {case_data['baseline']['extracted_answer']}\n"
        details_text += f"Variant Answer: {variant_info['extracted_answer']}"
        
        axes[1, 2].text(0.05, 0.5, details_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center', wrap=True)
        axes[1, 2].set_title('Question Details')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Attention Analysis: {case_data["group"]["baseline_question"][:100]}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def analyze_all_cases(self):
        """Run attention analysis on all changed answer cases"""
        # Load cases
        cases = self.load_changed_answer_cases()
        logger.info(f"Loaded {len(cases)} cases with changed answers")
        
        all_results = []
        
        for i, case in enumerate(tqdm(cases, desc="Analyzing attention changes")):
            logger.info(f"\nAnalyzing case {i+1}: {case['group']['baseline_question']}")
            
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
            
            # Analyze each changed variant
            for variant in case['changed_variants']:
                logger.info(f"  Extracting attention for variant: {variant['strategy']}")
                
                variant_result = self.extract_attention_for_question(
                    image, variant['question']
                )
                
                # Compare attentions
                if baseline_result['attention_grid'] and variant_result['attention_grid']:
                    comparison = self.compare_attention_maps(
                        baseline_result['attention_grid'],
                        variant_result['attention_grid']
                    )
                    
                    # Create visualization
                    fig = self.visualize_attention_comparison(
                        case, 
                        baseline_result['attention_grid'],
                        variant_result['attention_grid'],
                        variant
                    )
                    
                    # Save visualization
                    viz_path = self.dirs['comparisons'] / f"case_{i:02d}_variant_{variant['variant_index']}.png"
                    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    case_results['variants'].append({
                        'variant_index': variant['variant_index'],
                        'strategy': variant['strategy'],
                        'question': variant['question'],
                        'answer': variant['extracted_answer'],
                        'attention_metrics': comparison['metrics'] if comparison else None
                    })
            
            all_results.append(case_results)
        
        # Save results
        self.save_results(all_results)
        return all_results
    
    def save_results(self, results):
        """Save analysis results"""
        # Save as pickle for full data
        with open(self.dirs['statistics'] / 'attention_change_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary statistics
        summary = self.compute_summary_statistics(results)
        
        # Save summary
        with open(self.dirs['statistics'] / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        
        # Generate report
        self.generate_report(results, summary)
    
    def compute_summary_statistics(self, results):
        """Compute summary statistics from results"""
        all_js_divs = []
        all_correlations = []
        all_l2_dists = []
        all_cosine_sims = []
        
        for case in results:
            for variant in case.get('variants', []):
                if variant.get('attention_metrics'):
                    metrics = variant['attention_metrics']
                    all_js_divs.append(metrics['js_divergence'])
                    all_correlations.append(metrics['correlation'])
                    all_l2_dists.append(metrics['l2_distance'])
                    all_cosine_sims.append(metrics['cosine_similarity'])
        
        summary = {
            'num_cases_analyzed': len(results),
            'total_comparisons': len(all_js_divs),
            'js_divergence': {
                'mean': np.mean(all_js_divs) if all_js_divs else 0,
                'std': np.std(all_js_divs) if all_js_divs else 0,
                'min': np.min(all_js_divs) if all_js_divs else 0,
                'max': np.max(all_js_divs) if all_js_divs else 0
            },
            'correlation': {
                'mean': np.mean(all_correlations) if all_correlations else 0,
                'std': np.std(all_correlations) if all_correlations else 0,
                'min': np.min(all_correlations) if all_correlations else 0,
                'max': np.max(all_correlations) if all_correlations else 0
            },
            'l2_distance': {
                'mean': np.mean(all_l2_dists) if all_l2_dists else 0,
                'std': np.std(all_l2_dists) if all_l2_dists else 0
            },
            'cosine_similarity': {
                'mean': np.mean(all_cosine_sims) if all_cosine_sims else 0,
                'std': np.std(all_cosine_sims) if all_cosine_sims else 0
            }
        }
        
        return summary
    
    def generate_report(self, results, summary):
        """Generate analysis report"""
        report = []
        report.append("="*80)
        report.append("ATTENTION CHANGE ANALYSIS FOR PHRASING VARIATIONS")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("SUMMARY STATISTICS")
        report.append("-"*40)
        report.append(f"Cases analyzed: {summary['num_cases_analyzed']}")
        report.append(f"Total comparisons: {summary['total_comparisons']}")
        report.append("")
        
        report.append("ATTENTION DIVERGENCE METRICS")
        report.append("-"*40)
        report.append(f"JS Divergence: {summary['js_divergence']['mean']:.3f} ± {summary['js_divergence']['std']:.3f}")
        report.append(f"  Range: [{summary['js_divergence']['min']:.3f}, {summary['js_divergence']['max']:.3f}]")
        report.append(f"Correlation: {summary['correlation']['mean']:.3f} ± {summary['correlation']['std']:.3f}")
        report.append(f"  Range: [{summary['correlation']['min']:.3f}, {summary['correlation']['max']:.3f}]")
        report.append(f"Cosine Similarity: {summary['cosine_similarity']['mean']:.3f} ± {summary['cosine_similarity']['std']:.3f}")
        report.append("")
        
        report.append("CASE DETAILS")
        report.append("-"*40)
        
        for i, case in enumerate(results[:5]):  # Show first 5 cases
            report.append(f"\nCase {i+1}: {case['baseline_question']}")
            report.append(f"  Ground Truth: {case['ground_truth']}")
            report.append(f"  Baseline Answer: {case['baseline_answer']}")
            
            for variant in case.get('variants', []):
                report.append(f"\n  Variant ({variant['strategy']}):")
                report.append(f"    Question: {variant['question'][:80]}...")
                report.append(f"    Answer Changed: {case['baseline_answer']} → {variant['answer']}")
                
                if variant.get('attention_metrics'):
                    m = variant['attention_metrics']
                    report.append(f"    JS Divergence: {m['js_divergence']:.3f}")
                    report.append(f"    Correlation: {m['correlation']:.3f}")
        
        report.append("")
        report.append("="*80)
        report_text = "\n".join(report)
        
        with open(self.dirs['statistics'] / 'report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


def main():
    """Main execution function"""
    import logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("Starting attention change analysis for phrasing variations")
    
    # Setup GPU
    device = setup_gpu(min_free_gb=15.0)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading MedGemma model...")
    model, processor = load_model_enhanced(device=device)
    
    # Create analyzer
    analyzer = AttentionChangeAnalyzer(model, processor, OUTPUT_DIR)
    
    # Run analysis
    logger.info("Running attention analysis on changed answer cases...")
    results = analyzer.analyze_all_cases()
    
    logger.info(f"Analysis complete! Results saved to {OUTPUT_DIR}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    results = main()