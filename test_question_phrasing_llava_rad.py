#!/usr/bin/env python3
"""
Question Phrasing Sensitivity Analysis for LLaVA-Rad
Tests linguistic robustness of Microsoft's LLaVA-Rad model
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import torch
import logging
from PIL import Image

# Import LLaVA-Rad visualizer
from llava_rad_visualizer import LLaVARadVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_rad_phrasing_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVARadPhrasingAnalyzer:
    """Analyze LLaVA-Rad's sensitivity to question phrasing"""
    
    def __init__(self, visualizer, output_dir="llava_rad_phrasing_results"):
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def analyze_question_group(self, question_group_df, image_path):
        """Analyze all phrasing variants for a question"""
        results = {
            'baseline_question': None,
            'image_path': str(image_path),
            'ground_truth': None,
            'variants': [],
            'consistency_rate': 0.0,
            'answer_distribution': {}
        }
        
        # Load image once
        image = Image.open(image_path).convert('RGB')
        
        answers = []
        
        for idx, row in question_group_df.iterrows():
            question = row['question']  # Changed from 'questions' to 'question'
            variant_index = row.get('variant_index', idx)
            strategy = row.get('strategy', 'unknown')  # Changed from 'strategies' to 'strategy'
            
            # Set baseline info
            if variant_index == 0:
                results['baseline_question'] = question
                results['ground_truth'] = row.get('answer', 'unknown')  # Changed from 'ground_truth_answer' to 'answer'
            
            # Generate answer
            try:
                analysis = self.visualizer.generate_with_attention(image, question)
                raw_answer = analysis['answer']
                extracted_answer = self.visualizer.extract_answer(raw_answer)
                
                variant_result = {
                    'variant_index': variant_index,
                    'question': question,
                    'strategy': strategy,
                    'raw_answer': raw_answer[:200],  # Truncate for storage
                    'extracted_answer': extracted_answer
                }
                
                results['variants'].append(variant_result)
                answers.append(extracted_answer)
                
                logger.info(f"  Variant {variant_index}: {extracted_answer} - {strategy[:30]}")
                
            except Exception as e:
                logger.error(f"  Error with variant {variant_index}: {e}")
                variant_result = {
                    'variant_index': variant_index,
                    'question': question,
                    'strategy': strategy,
                    'raw_answer': f"ERROR: {str(e)}",
                    'extracted_answer': 'error'
                }
                results['variants'].append(variant_result)
        
        # Calculate consistency
        if answers:
            # Count distribution
            for ans in answers:
                results['answer_distribution'][ans] = results['answer_distribution'].get(ans, 0) + 1
            
            # Consistency is how often all variants agree
            if len(set(answers)) == 1:
                results['consistency_rate'] = 1.0
            else:
                # Proportion of variants that match the baseline
                baseline_answer = results['variants'][0]['extracted_answer'] if results['variants'] else None
                if baseline_answer:
                    matching = sum(1 for a in answers if a == baseline_answer)
                    results['consistency_rate'] = matching / len(answers)
        
        return results
    
    def run_analysis(self, csv_path, max_groups=None):
        """Run complete phrasing analysis"""
        logger.info(f"Loading questions from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} question variants")
        
        # Group by base question
        grouped = df.groupby('baseline_question')
        
        all_results = []
        group_count = 0
        
        for group_name, group_df in tqdm(grouped, desc="Analyzing question groups"):
            if max_groups and group_count >= max_groups:
                break
            
            # Get image path - add base path if needed
            image_path = group_df.iloc[0]['image_path']
            if not Path(image_path).is_absolute():
                image_path = f"/home/bsada1/mimic_cxr_hundred_vqa/{image_path}"
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            logger.info(f"\nGroup {group_count + 1}: {group_df.iloc[0]['question'][:50]}...")
            
            # Analyze this question group
            group_results = self.analyze_question_group(group_df, image_path)
            all_results.append(group_results)
            
            # Save progress
            if (group_count + 1) % 5 == 0:
                self.save_results(all_results)
            
            group_count += 1
        
        # Final save
        self.save_results(all_results)
        self.generate_report(all_results)
        
        return all_results
    
    def save_results(self, results):
        """Save analysis results"""
        # Save detailed results as JSON
        with open(self.output_dir / 'llava_rad_detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for group in results:
            for variant in group['variants']:
                summary_data.append({
                    'baseline_question': group['baseline_question'],
                    'variant_index': variant['variant_index'],
                    'strategy': variant['strategy'],
                    'question': variant['question'],
                    'answer': variant['extracted_answer'],
                    'ground_truth': group['ground_truth'],
                    'consistency_rate': group['consistency_rate']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / 'llava_rad_phrasing_summary.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def generate_report(self, results):
        """Generate analysis report"""
        report = []
        report.append("="*80)
        report.append("LLaVA-Rad Question Phrasing Sensitivity Analysis")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        total_groups = len(results)
        
        if total_groups == 0:
            report.append("No question groups were successfully analyzed.")
            report_text = "\n".join(report)
            with open(self.output_dir / 'llava_rad_phrasing_report.txt', 'w') as f:
                f.write(report_text)
            print(report_text)
            return report_text
        
        consistency_rates = [r['consistency_rate'] for r in results]
        
        report.append(f"Total question groups analyzed: {total_groups}")
        report.append(f"Mean consistency rate: {np.mean(consistency_rates):.3f} ± {np.std(consistency_rates):.3f}")
        
        # Fully consistent groups
        fully_consistent = sum(1 for rate in consistency_rates if rate == 1.0)
        report.append(f"Fully consistent groups: {fully_consistent}/{total_groups} ({fully_consistent/total_groups*100:.1f}%)")
        
        # Groups with variations
        with_variations = total_groups - fully_consistent
        report.append(f"Groups with answer variations: {with_variations}/{total_groups} ({with_variations/total_groups*100:.1f}%)\n")
        
        # Most inconsistent questions
        report.append("Most Inconsistent Questions:")
        sorted_results = sorted(results, key=lambda x: x['consistency_rate'])
        for i, group in enumerate(sorted_results[:5]):
            report.append(f"{i+1}. {group['baseline_question'][:60]}...")
            report.append(f"   Consistency: {group['consistency_rate']:.2f}")
            report.append(f"   Answer distribution: {group['answer_distribution']}\n")
        
        # Strategy performance
        strategy_accuracy = {}
        strategy_counts = {}
        
        for group in results:
            ground_truth = group['ground_truth']
            for variant in group['variants']:
                strategy = variant['strategy']
                if strategy not in strategy_accuracy:
                    strategy_accuracy[strategy] = []
                    strategy_counts[strategy] = 0
                
                is_correct = (variant['extracted_answer'] == ground_truth)
                strategy_accuracy[strategy].append(is_correct)
                strategy_counts[strategy] += 1
        
        report.append("\nPerformance by Phrasing Strategy:")
        strategy_summary = []
        for strategy, correct_list in strategy_accuracy.items():
            if correct_list:
                accuracy = sum(correct_list) / len(correct_list)
                strategy_summary.append((strategy, accuracy, len(correct_list)))
        
        # Sort by accuracy
        strategy_summary.sort(key=lambda x: x[1], reverse=True)
        
        for strategy, accuracy, count in strategy_summary[:10]:
            report.append(f"  {strategy[:40]:<40} Acc: {accuracy:.3f} (n={count})")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'llava_rad_phrasing_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main execution"""
    logger.info("Starting LLaVA-Rad question phrasing analysis")
    
    # Setup LLaVA-Rad
    visualizer = LLaVARadVisualizer()
    visualizer.load_model(load_in_8bit=True)
    
    # Create analyzer
    analyzer = LLaVARadPhrasingAnalyzer(visualizer)
    
    # Run analysis
    csv_path = "/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv"
    
    # Run on all groups
    results = analyzer.run_analysis(csv_path, max_groups=None)  # None means all groups
    
    logger.info(f"Analysis complete! Analyzed {len(results)} question groups")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    results = main()