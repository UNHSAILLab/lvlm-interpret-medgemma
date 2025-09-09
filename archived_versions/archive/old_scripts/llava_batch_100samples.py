#!/usr/bin/env python3
"""
Batch analysis of 100 MIMIC-CXR samples using LLaVA
Parallel to batch_analysis_100samples.py for MedGemma
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datetime import datetime

# Import LLaVA visualizer
from llava_rad_visualizer import LLaVARadVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_batch_100samples.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVABatchAnalyzer:
    """Batch analysis for LLaVA on MIMIC-CXR 100 samples"""
    
    def __init__(self, visualizer, output_dir="llava_100samples_results"):
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'visualizations': self.output_dir / 'visualizations',
            'attention_maps': self.output_dir / 'attention_maps',
            'statistics': self.output_dir / 'statistics'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def run_batch_analysis(self, csv_path):
        """Run analysis on 100 samples"""
        logger.info(f"Loading samples from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        results = []
        correct_count = 0
        
        # Process each sample
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            try:
                # Build image path
                img_path = f"/home/bsada1/mimic_cxr_hundred_vqa/{row['dicom_id']}.jpg"
                
                if not Path(img_path).exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Generate answer with attention
                result = self.visualizer.generate_with_attention(
                    image, 
                    row['question'],
                    max_new_tokens=100
                )
                
                # Extract answer
                extracted_answer = self.visualizer.extract_answer(result['answer'])
                
                # Check correctness
                is_correct = extracted_answer == row['answer']
                if is_correct:
                    correct_count += 1
                
                # Compute attention metrics
                attention_metrics = {}
                if result['visual_attention'] is not None:
                    attention_metrics = self.visualizer.compute_attention_metrics(
                        result['visual_attention']
                    )
                    
                    # Save attention map
                    np.save(
                        self.dirs['attention_maps'] / f"{row['dicom_id']}_attention.npy",
                        result['visual_attention']
                    )
                
                # Store result
                sample_result = {
                    'index': idx,
                    'dicom_id': row['dicom_id'],
                    'question': row['question'],
                    'question_type': row.get('question_type', 'unknown'),
                    'ground_truth': row['answer'],
                    'model_answer': extracted_answer,
                    'raw_answer': result['answer'][:500],  # Store more of the answer
                    'correct': is_correct,
                    'attention_metrics': attention_metrics
                }
                
                results.append(sample_result)
                
                # Log progress every 10 samples
                if (idx + 1) % 10 == 0:
                    current_accuracy = correct_count / (idx + 1) * 100
                    logger.info(f"Processed {idx+1}/{len(df)} - Current accuracy: {current_accuracy:.1f}%")
                    
                    # Save interim results
                    self.save_interim_results(results)
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate final statistics
        final_accuracy = correct_count / len(results) * 100 if results else 0
        
        # Compute statistics by question type
        stats_by_type = self.compute_statistics_by_type(results)
        
        # Generate final report
        self.generate_report(results, final_accuracy, stats_by_type)
        
        # Save all results
        self.save_final_results(results, final_accuracy, stats_by_type)
        
        logger.info(f"Analysis complete! Final accuracy: {final_accuracy:.2f}%")
        
        return results, final_accuracy
    
    def save_interim_results(self, results):
        """Save interim results during processing"""
        with open(self.dirs['statistics'] / 'interim_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
    
    def compute_statistics_by_type(self, results):
        """Compute statistics by question type"""
        df_results = pd.DataFrame(results)
        
        if 'question_type' not in df_results.columns:
            return {}
        
        stats = {}
        for qtype in df_results['question_type'].unique():
            type_df = df_results[df_results['question_type'] == qtype]
            
            stats[qtype] = {
                'count': len(type_df),
                'correct': int(type_df['correct'].sum()),
                'accuracy': float(type_df['correct'].mean() * 100),
                'samples': len(type_df)
            }
        
        return stats
    
    def generate_report(self, results, accuracy, stats_by_type):
        """Generate analysis report"""
        report = []
        report.append("="*80)
        report.append("LLaVA-Rad Batch Analysis Report - 100 MIMIC-CXR Samples")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append(f"Total samples analyzed: {len(results)}")
        report.append(f"Overall accuracy: {accuracy:.2f}%\n")
        
        if stats_by_type:
            report.append("Performance by Question Type:")
            for qtype, stats in sorted(stats_by_type.items(), 
                                      key=lambda x: x[1]['accuracy'], 
                                      reverse=True):
                report.append(f"  {qtype}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['count']})")
        
        # Analyze errors
        errors = [r for r in results if not r['correct']]
        if errors:
            report.append(f"\nTotal errors: {len(errors)}")
            report.append("\nSample errors:")
            for err in errors[:5]:
                report.append(f"  Q: {err['question'][:50]}...")
                report.append(f"  GT: {err['ground_truth']}, Model: {err['model_answer']}")
        
        # Attention metrics summary
        if results and 'attention_metrics' in results[0] and results[0]['attention_metrics']:
            report.append("\nAttention Metrics Summary:")
            metrics_df = pd.DataFrame([r['attention_metrics'] for r in results 
                                      if r.get('attention_metrics')])
            
            for metric in metrics_df.columns:
                if metric in ['max_attention', 'mean_attention', 'entropy', 'focus_score']:
                    mean_val = metrics_df[metric].mean()
                    std_val = metrics_df[metric].std()
                    report.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.output_dir / 'batch_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def save_final_results(self, results, accuracy, stats_by_type):
        """Save all final results"""
        # Save detailed results
        with open(self.dirs['statistics'] / 'detailed_results.json', 'w') as f:
            json.dump({
                'results': results,
                'summary': {
                    'total_samples': len(results),
                    'accuracy': accuracy,
                    'stats_by_type': stats_by_type,
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=2, default=self.json_serializer)
        
        # Save summary CSV
        df_results = pd.DataFrame(results)
        if len(df_results) > 0:
            summary_df = df_results[['dicom_id', 'question', 'ground_truth', 
                                    'model_answer', 'correct']]
            summary_df.to_csv(self.output_dir / 'results_summary.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    @staticmethod
    def json_serializer(obj):
        """JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def main():
    """Main execution"""
    logger.info("Starting LLaVA-Rad batch analysis on 100 MIMIC-CXR samples")
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    
    # Initialize visualizer
    visualizer = LLaVARadVisualizer()
    visualizer.load_model(load_in_8bit=True)
    
    # Create analyzer
    analyzer = LLaVABatchAnalyzer(visualizer)
    
    # Run analysis on 100 samples
    csv_path = "/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample.csv"
    results, accuracy = analyzer.run_batch_analysis(csv_path)
    
    logger.info(f"Batch analysis complete! Processed {len(results)} samples")
    logger.info(f"Final accuracy: {accuracy:.2f}%")
    
    # Clean up
    torch.cuda.empty_cache()
    
    return results, accuracy


if __name__ == "__main__":
    results, accuracy = main()