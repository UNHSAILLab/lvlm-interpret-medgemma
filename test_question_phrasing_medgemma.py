#!/usr/bin/env python3
"""
Test MedGemma's sensitivity to question phrasing variations
Analyzes how different phrasings of the same medical question affect model responses
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import logging
from collections import defaultdict

# Add path for imports
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from medgemma_launch_mimic_fixed import setup_gpu, load_model_enhanced

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_phrasing_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
QUESTIONS_CSV = "/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv"
IMAGE_BASE_PATH = "/home/bsada1/mimic_cxr_hundred_vqa"
OUTPUT_DIR = "question_phrasing_results"

class QuestionPhrasingAnalyzer:
    """Analyze MedGemma responses to different question phrasings"""
    
    def __init__(self, model, processor, output_dir="question_phrasing_results"):
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = next(model.parameters()).device
        
        # Results storage
        self.results = []
        self.phrasing_groups = defaultdict(list)
        
    def extract_answer(self, text):
        """Extract yes/no answer from generated text"""
        text_lower = text.lower().strip()
        
        # Direct yes/no at start
        if text_lower.startswith('yes'):
            return 'yes'
        elif text_lower.startswith('no'):
            return 'no'
        
        # Look in first sentence
        first_sentence = text_lower.split('.')[0] if '.' in text_lower else text_lower
        
        # No indicators (check first as they're often more explicit)
        no_indicators = ['no', 'there is no', 'there are no', 'absent', 'not present',
                        'not evident', 'not visible', 'not detected', 'not observed',
                        'does not show', 'negative for', 'without', 'no evidence']
        
        for indicator in no_indicators:
            if indicator in first_sentence:
                return 'no'
        
        # Yes indicators
        yes_indicators = ['yes', 'there is', 'there are', 'present', 'evident', 
                         'visible', 'detected', 'observed', 'shows', 'demonstrates',
                         'consistent with', 'suggestive of', 'positive for']
        
        for indicator in yes_indicators:
            if indicator in first_sentence:
                return 'yes'
        
        # Check first 100 chars
        text_start = text_lower[:100]
        if 'yes' in text_start:
            return 'yes'
        elif 'no' in text_start:
            return 'no'
        
        return 'uncertain'
    
    def get_model_response(self, image, question):
        """Get model's response to a question"""
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
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0  # Deterministic for consistency
            )
        
        # Extract text
        generated_ids = outputs[0][len(inputs['input_ids'][0]):]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def analyze_question_group(self, question_group_df):
        """Analyze all phrasings for a single question"""
        # Get the baseline question info
        baseline_row = question_group_df[question_group_df['strategy'] == 'baseline'].iloc[0]
        image_path = Path(IMAGE_BASE_PATH) / baseline_row['image_path']
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Load image once for all variants
        image = Image.open(image_path).convert('RGB')
        
        group_results = {
            'study_id': baseline_row['study_id'],
            'baseline_question': baseline_row['baseline_question'],
            'ground_truth': baseline_row['answer'],
            'image_path': str(image_path),
            'variants': []
        }
        
        # Test each phrasing variant
        for _, row in question_group_df.iterrows():
            question = row['question']
            variant_index = row['variant_index']
            strategy = row['strategy']
            
            # Get model response
            generated_text = self.get_model_response(image, question)
            extracted_answer = self.extract_answer(generated_text)
            
            variant_result = {
                'variant_index': variant_index,
                'strategy': strategy,
                'question': question,
                'generated_text': generated_text,
                'extracted_answer': extracted_answer,
                'is_correct': extracted_answer == row['answer'].lower(),
            }
            
            group_results['variants'].append(variant_result)
            
            logger.info(f"  Variant {variant_index} ({strategy}): {extracted_answer} "
                       f"({'✓' if variant_result['is_correct'] else '✗'})")
        
        # Calculate consistency metrics
        answers = [v['extracted_answer'] for v in group_results['variants']]
        group_results['all_consistent'] = len(set(answers)) == 1
        group_results['majority_answer'] = max(set(answers), key=answers.count)
        group_results['consistency_rate'] = answers.count(group_results['majority_answer']) / len(answers)
        
        # Check if baseline is correct
        baseline_variant = [v for v in group_results['variants'] if v['variant_index'] == 0][0]
        group_results['baseline_correct'] = baseline_variant['is_correct']
        
        return group_results
    
    def run_analysis(self, num_groups=None):
        """Run analysis on question groups"""
        # Load questions
        df = pd.read_csv(QUESTIONS_CSV)
        logger.info(f"Loaded {len(df)} question variants")
        
        # Group by baseline question
        grouped = df.groupby('baseline_question')
        
        if num_groups:
            grouped = list(grouped)[:num_groups]
        else:
            grouped = list(grouped)
        
        logger.info(f"Analyzing {len(grouped)} question groups")
        
        # Process each group
        for i, (baseline_q, group_df) in enumerate(tqdm(grouped, desc="Analyzing question groups")):
            logger.info(f"\nGroup {i+1}: {baseline_q}")
            
            group_results = self.analyze_question_group(group_df)
            
            if group_results:
                self.results.append(group_results)
                self.phrasing_groups[baseline_q] = group_results
                
                # Log summary
                logger.info(f"  Consistency: {group_results['consistency_rate']:.1%}")
                logger.info(f"  Baseline correct: {group_results['baseline_correct']}")
        
        # Save results
        self.save_results()
        self.generate_summary()
        
        return self.results
    
    def save_results(self):
        """Save detailed results"""
        # Save full results as JSON
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create flattened CSV for analysis
        rows = []
        for group in self.results:
            for variant in group['variants']:
                row = {
                    'study_id': group['study_id'],
                    'baseline_question': group['baseline_question'],
                    'ground_truth': group['ground_truth'],
                    'variant_index': variant['variant_index'],
                    'strategy': variant['strategy'],
                    'question': variant['question'],
                    'generated_text': variant['generated_text'],
                    'extracted_answer': variant['extracted_answer'],
                    'is_correct': variant['is_correct'],
                    'group_consistent': group['all_consistent'],
                    'consistency_rate': group['consistency_rate']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'phrasing_analysis.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def generate_summary(self):
        """Generate summary statistics and report"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        summary = {
            'total_groups': len(self.results),
            'total_variants_tested': sum(len(g['variants']) for g in self.results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Overall consistency
        consistency_rates = [g['consistency_rate'] for g in self.results]
        summary['overall_consistency'] = {
            'mean': np.mean(consistency_rates),
            'std': np.std(consistency_rates),
            'min': np.min(consistency_rates),
            'max': np.max(consistency_rates),
            'fully_consistent_groups': sum(1 for g in self.results if g['all_consistent'])
        }
        
        # Accuracy by strategy
        strategy_accuracy = defaultdict(list)
        for group in self.results:
            for variant in group['variants']:
                strategy_accuracy[variant['strategy']].append(variant['is_correct'])
        
        summary['accuracy_by_strategy'] = {
            strategy: np.mean(correct_list)
            for strategy, correct_list in strategy_accuracy.items()
        }
        
        # Baseline vs variants
        baseline_correct = [g['baseline_correct'] for g in self.results]
        summary['baseline_accuracy'] = np.mean(baseline_correct)
        
        # Cases where phrasing changed answer
        changed_answer_groups = []
        for group in self.results:
            if not group['all_consistent']:
                changed_answer_groups.append({
                    'baseline_question': group['baseline_question'],
                    'ground_truth': group['ground_truth'],
                    'consistency_rate': group['consistency_rate'],
                    'answers': {v['strategy']: v['extracted_answer'] 
                              for v in group['variants']}
                })
        
        summary['groups_with_changed_answers'] = changed_answer_groups
        summary['num_groups_changed'] = len(changed_answer_groups)
        
        # Save summary
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        
        # Generate text report
        report = []
        report.append("="*80)
        report.append("QUESTION PHRASING SENSITIVITY ANALYSIS - MEDGEMMA")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total question groups analyzed: {summary['total_groups']}")
        report.append(f"Total variants tested: {summary['total_variants_tested']}")
        report.append("")
        
        report.append("CONSISTENCY ANALYSIS")
        report.append("-"*40)
        report.append(f"Mean consistency rate: {summary['overall_consistency']['mean']:.1%}")
        report.append(f"Std deviation: {summary['overall_consistency']['std']:.1%}")
        report.append(f"Fully consistent groups: {summary['overall_consistency']['fully_consistent_groups']}/{summary['total_groups']}")
        report.append(f"Groups with changed answers: {summary['num_groups_changed']}")
        report.append("")
        
        report.append("ACCURACY BY PHRASING STRATEGY")
        report.append("-"*40)
        for strategy, accuracy in summary['accuracy_by_strategy'].items():
            report.append(f"{strategy}: {accuracy:.1%}")
        report.append("")
        
        report.append("BASELINE ACCURACY")
        report.append("-"*40)
        report.append(f"Baseline questions accuracy: {summary['baseline_accuracy']:.1%}")
        report.append("")
        
        if changed_answer_groups:
            report.append("EXAMPLES OF INCONSISTENT RESPONSES")
            report.append("-"*40)
            for i, group in enumerate(changed_answer_groups[:5]):  # Show first 5
                report.append(f"\nExample {i+1}:")
                report.append(f"Question: {group['baseline_question']}")
                report.append(f"Ground truth: {group['ground_truth']}")
                report.append(f"Consistency: {group['consistency_rate']:.1%}")
                report.append("Answers by strategy:")
                for strategy, answer in group['answers'].items():
                    report.append(f"  - {strategy}: {answer}")
        
        report.append("")
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return summary


def main():
    """Main execution function"""
    logger.info("Starting Question Phrasing Sensitivity Analysis")
    
    # Setup GPU
    device = setup_gpu(min_free_gb=15.0)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading MedGemma model...")
    model, processor = load_model_enhanced(device=device)
    
    # Create analyzer
    analyzer = QuestionPhrasingAnalyzer(model, processor, OUTPUT_DIR)
    
    # Run analysis on all groups
    logger.info("Running phrasing analysis on ALL question groups...")
    results = analyzer.run_analysis()  # No limit - analyze all groups
    
    logger.info(f"Analysis complete! Results saved to {OUTPUT_DIR}")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    results = main()