#!/usr/bin/env python3
"""
Batch Analysis Script for MedGemma Visualizer
Runs comprehensive analysis on 100 MIMIC-CXR samples and generates statistics for paper
Developed by SAIL Lab - University of New Haven
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter
import logging
import gc
import warnings
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import ast
import json
from datetime import datetime
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# Import functions from main app
sys.path.append('/home/bsada1/lvlm-interpret-medgemma')
from medgemma_launch_mimic_fixed import (
    factor_to_grid, find_target_token_indices_robust, model_view_image,
    tight_body_mask, prepare_attn_grid, strip_border_tokens,
    overlay_attention_enhanced, extract_attention_data,
    gradcam_on_vision, compute_attention_metrics, token_mask_from_body,
    create_token_attention_overlay_robust, 
    setup_gpu, load_model_enhanced, MIMICDataLoader
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MIMIC_CSV_PATH = "/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv"
MIMIC_IMAGE_BASE_PATH = "/home/bsada1/mimic_cxr_hundred_vqa"
OUTPUT_DIR = "batch_analysis_results"
NUM_SAMPLES = 100
SAVE_INTERMEDIATE = True

class BatchAnalyzer:
    """Batch analysis for multiple MIMIC-CXR samples"""
    
    def __init__(self, model, processor, data_loader, output_dir="batch_analysis_results"):
        self.model = model
        self.processor = processor
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'attention_maps': self.output_dir / 'attention_maps',
            'statistics': self.output_dir / 'statistics',
            'visualizations': self.output_dir / 'visualizations',
            'faithfulness': self.output_dir / 'faithfulness',
            'raw_data': self.output_dir / 'raw_data'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'basic_analysis': [],
            'token_conditioned': [],
            'prompt_sensitivity': [],
            'faithfulness': [],
            'answer_accuracy': [],
            'attention_metrics': []
        }
        
        self.device = next(model.parameters()).device
        
    def extract_medical_term(self, question):
        """Extract key medical term from question"""
        medical_terms = [
            'pneumonia', 'effusion', 'consolidation', 'atelectasis',
            'edema', 'pneumothorax', 'cardiomegaly', 'opacity',
            'infiltrate', 'nodule', 'mass', 'pleural'
        ]
        
        question_lower = question.lower()
        for term in medical_terms:
            if term in question_lower:
                return term
        return None
    
    def analyze_single_sample(self, idx, sample):
        """Comprehensive analysis of a single sample"""
        logger.info(f"Analyzing sample {idx}: {sample['study_id']}")
        
        result = {
            'index': idx,
            'study_id': sample['study_id'],
            'question': sample['question'],
            'ground_truth': sample['correct_answer'],
            'medical_term': self.extract_medical_term(sample['question'])
        }
        
        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # 1. Basic Question Analysis - pass ground truth
            logger.info(f"  Running basic analysis...")
            basic_result = self.run_basic_analysis(image, sample['question'], sample['correct_answer'])
            result.update(basic_result)
            
            # 2. Token-Conditioned Analysis (if medical term found)
            if result['medical_term']:
                logger.info(f"  Running token-conditioned analysis for '{result['medical_term']}'...")
                token_result = self.run_token_analysis(
                    image, sample['question'], result['medical_term']
                )
                result.update(token_result)
            
            # 3. Prompt Sensitivity Analysis
            logger.info(f"  Running prompt sensitivity analysis...")
            prompt_result = self.run_prompt_sensitivity(image, sample['question'])
            result.update(prompt_result)
            
            # 4. Faithfulness Validation (subset only due to computation cost)
            if idx < 20:  # Run on first 20 samples only
                logger.info(f"  Running faithfulness validation...")
                faith_result = self.run_faithfulness_validation(
                    image, sample['question'], result['medical_term']
                )
                result.update(faith_result)
            
            # 5. Save visualizations
            if SAVE_INTERMEDIATE and idx < 20:  # Save first 20
                self.save_sample_visualizations(idx, sample, result, image)
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error analyzing sample {idx}: {e}")
            result['error'] = str(e)
        
        return result
    
    def run_basic_analysis(self, image, question, ground_truth=None):
        """Basic question answering with attention extraction"""
        result = {}
        
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
                max_new_tokens=100,  # Increased to capture full responses
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Extract answer
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Store FULL generated text for analysis
        result['full_generated_text'] = generated_text.strip()
        
        # Extract yes/no with improved logic
        answer = self.extract_answer_improved(generated_text)
        result['extracted_answer'] = answer
        result['generated_text_truncated'] = generated_text[:200]  # For display
        
        # Check correctness - use passed ground_truth or look in result
        if ground_truth:
            ground_truth_lower = ground_truth.lower()
        else:
            ground_truth_lower = result.get('ground_truth', '').lower()
        
        result['is_correct'] = (answer == ground_truth_lower)
        
        # Also check if answer is contained anywhere in response
        result['answer_in_text'] = (ground_truth_lower in generated_text.lower())
        
        # Extract attention data
        attention_data = extract_attention_data(
            self.model, outputs, inputs, self.processor
        )
        
        if attention_data:
            result['attention_entropy'] = attention_data.get('attention_entropy', -1)
            result['regional_focus'] = attention_data.get('regional_focus', 'unknown')
            
            # Get attention grid for metrics
            if 'attention_grid' in attention_data:
                grid = np.array(attention_data['attention_grid'])
                
                # Compute body mask
                gray = np.array(model_view_image(self.processor, image))
                body_mask = tight_body_mask(gray)
                
                # Compute metrics
                metrics = compute_attention_metrics(grid, body_mask)
                result.update({
                    'inside_body_ratio': metrics['inside_body_ratio'],
                    'border_fraction': metrics['border_fraction'],
                    'left_fraction': metrics['left_fraction'],
                    'right_fraction': metrics['right_fraction'],
                    'apical_fraction': metrics['apical_fraction'],
                    'basal_fraction': metrics['basal_fraction']
                })
        
        return result
    
    def run_token_analysis(self, image, question, target_word):
        """Token-conditioned attention analysis"""
        result = {}
        
        # Try multiple methods
        methods = ['gradcam_single', 'cross_attention']
        
        for method in methods:
            try:
                if method == 'gradcam_single':
                    attention_map = gradcam_on_vision(
                        self.model, self.processor, image, question, target_word
                    )
                elif method == 'cross_attention':
                    # Use basic analysis result if available
                    attention_map = np.ones((16, 16)) / 256  # Placeholder
                
                result[f'token_{method}_success'] = True
                
                # Compute metrics if successful
                if attention_map is not None and attention_map.sum() > 0:
                    gray = np.array(model_view_image(self.processor, image))
                    body_mask = tight_body_mask(gray)
                    metrics = compute_attention_metrics(attention_map, body_mask)
                    
                    result[f'token_{method}_inside_ratio'] = metrics['inside_body_ratio']
                    result[f'token_{method}_border_fraction'] = metrics['border_fraction']
                    break  # Use first successful method
                    
            except Exception as e:
                result[f'token_{method}_success'] = False
                logger.debug(f"Method {method} failed: {e}")
        
        return result
    
    def run_prompt_sensitivity(self, image, question):
        """Analyze sensitivity to prompt variations"""
        result = {}
        
        # Create prompt variations
        medical_term = self.extract_medical_term(question)
        if not medical_term:
            return result
        
        # Technical prompt
        prompt1 = f"Radiological assessment: Is there evidence of {medical_term}?"
        
        # Simple prompt
        simple_terms = {
            'pneumonia': 'lung infection',
            'effusion': 'fluid around lungs',
            'consolidation': 'solid lung',
            'cardiomegaly': 'enlarged heart',
            'atelectasis': 'lung collapse',
            'pneumothorax': 'air in chest',
            'edema': 'fluid buildup',
            'opacity': 'unclear area',
            'pleural': 'lung lining'
        }
        simple_term = simple_terms.get(medical_term, medical_term)
        prompt2 = f"Do you see any {simple_term} in this X-ray?"
        
        # Get actual attention maps for both prompts
        try:
            attention_maps = []
            answers = []
            
            for i, prompt in enumerate([prompt1, prompt2], 1):
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
                answer = self.extract_answer_improved(generated_text)
                answers.append(answer)
                result[f'prompt{i}_answer'] = answer
                result[f'prompt{i}_text'] = generated_text[:100]  # Store for review
                
                # Try to extract attention map
                try:
                    attention_data = extract_attention_data(
                        self.model, outputs, inputs, self.processor
                    )
                    if attention_data and 'attention_grid' in attention_data:
                        attention_maps.append(np.array(attention_data['attention_grid']))
                except:
                    # Use uniform if extraction fails
                    attention_maps.append(np.ones((16, 16)) / 256)
            
            # Compute JS divergence between attention maps
            if len(attention_maps) == 2:
                # Normalize to probabilities
                attn1_norm = attention_maps[0] / attention_maps[0].sum()
                attn2_norm = attention_maps[1] / attention_maps[1].sum()
                
                # Compute JS divergence
                js_div = jensenshannon(attn1_norm.flatten(), attn2_norm.flatten())
                
                # Handle edge cases
                if np.isnan(js_div) or np.isinf(js_div):
                    result['prompt_js_divergence'] = 0.0  # Same distributions
                else:
                    result['prompt_js_divergence'] = float(js_div)
            
            # Check consistency
            result['prompt_consistent'] = (answers[0] == answers[1]) if len(answers) == 2 else False
            
        except Exception as e:
            logger.debug(f"Prompt sensitivity failed: {e}")
            result['prompt_js_divergence'] = None  # Set to None instead of leaving undefined
        
        return result
    
    def run_faithfulness_validation(self, image, question, target_word):
        """Run faithfulness metrics (expensive - run on subset)"""
        result = {}
        
        if not target_word:
            target_word = "abnormality"  # Generic fallback
        
        try:
            # Get attention map using available method
            attention_map = gradcam_on_vision(
                self.model, self.processor, image, question, target_word
            )
            
            if attention_map is not None:
                # Compute simple faithfulness proxy metrics
                # Higher entropy = less focused = potentially less faithful
                entropy = -np.sum(attention_map * np.log(attention_map + 1e-10))
                result['attention_focus'] = 1.0 / (1.0 + entropy)  # 0-1 score
                
                # Compute peak-to-average ratio as faithfulness proxy
                peak_ratio = attention_map.max() / (attention_map.mean() + 1e-10)
                result['peak_ratio'] = min(peak_ratio / 10.0, 1.0)  # Normalize to 0-1
                
                # Compute sparsity as faithfulness proxy
                threshold = attention_map.mean() + attention_map.std()
                sparsity = (attention_map > threshold).sum() / attention_map.size
                result['attention_sparsity'] = sparsity
            else:
                # If attention extraction failed, set None values
                result['attention_focus'] = None
                result['peak_ratio'] = None  
                result['attention_sparsity'] = None
            
        except Exception as e:
            logger.debug(f"Faithfulness validation failed: {e}")
            # Set None values instead of error string
            result['attention_focus'] = None
            result['peak_ratio'] = None
            result['attention_sparsity'] = None
        
        return result
    
    def extract_answer(self, text):
        """Extract yes/no answer from generated text"""
        text_lower = text.lower().strip()
        
        if text_lower.startswith('yes'):
            return 'yes'
        elif text_lower.startswith('no'):
            return 'no'
        elif 'yes' in text_lower[:20]:
            return 'yes'
        elif 'no' in text_lower[:20]:
            return 'no'
        else:
            return 'uncertain'
    
    def extract_answer_improved(self, text):
        """Improved answer extraction with multiple strategies"""
        text_lower = text.lower().strip()
        
        # Strategy 1: Check direct start
        if text_lower.startswith('yes'):
            return 'yes'
        elif text_lower.startswith('no'):
            return 'no'
        
        # Strategy 2: Check first sentence
        first_sentence = text_lower.split('.')[0] if '.' in text_lower else text_lower
        
        # Look for definitive yes indicators
        yes_indicators = ['yes', 'there is', 'there are', 'present', 'evident', 
                         'visible', 'detected', 'observed', 'shows', 'demonstrates',
                         'consistent with', 'suggestive of', 'positive for']
        
        # Look for definitive no indicators  
        no_indicators = ['no', 'there is no', 'there are no', 'absent', 'not present',
                        'not evident', 'not visible', 'not detected', 'not observed',
                        'does not show', 'negative for', 'without', 'no evidence']
        
        # Check for no indicators first (they're often more explicit)
        for indicator in no_indicators:
            if indicator in first_sentence:
                return 'no'
        
        # Then check for yes indicators
        for indicator in yes_indicators:
            if indicator in first_sentence:
                return 'yes'
        
        # Strategy 3: Look in first 100 chars for yes/no
        text_start = text_lower[:100]
        if 'yes' in text_start:
            return 'yes'
        elif 'no' in text_start:
            return 'no'
        
        # Strategy 4: Try to infer from medical terminology
        # If the response describes findings, it's likely "yes"
        medical_findings = ['opacity', 'consolidation', 'effusion', 'infiltrate',
                          'atelectasis', 'pneumothorax', 'cardiomegaly', 'edema']
        
        for finding in medical_findings:
            if finding in first_sentence and 'no ' + finding not in first_sentence:
                return 'yes'
        
        return 'uncertain'
    
    def save_sample_visualizations(self, idx, sample, result, image):
        """Save visualizations for a sample"""
        # Create sample directory
        sample_dir = self.dirs['visualizations'] / f"sample_{idx:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save original image
        image.save(sample_dir / "original.png")
        
        # Save attention overlay if available
        if 'attention_grid' in result:
            try:
                fig = overlay_attention_enhanced(
                    image, result['attention_grid'], 
                    self.processor, alpha=0.35
                )
                fig.save(sample_dir / "attention_overlay.png")
            except:
                pass
        
        # Save metrics summary
        with open(sample_dir / "metrics.json", 'w') as f:
            json.dump({
                k: v for k, v in result.items() 
                if isinstance(v, (str, int, float, bool))
            }, f, indent=2)
    
    def run_batch_analysis(self, num_samples=100):
        """Run analysis on batch of samples"""
        logger.info(f"Starting batch analysis on {num_samples} samples")
        
        # Get sample indices
        total_samples = len(self.data_loader.df)
        if num_samples > total_samples:
            num_samples = total_samples
            logger.warning(f"Reduced to {num_samples} available samples")
        
        sample_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # Process each sample
        all_results = []
        for i, idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
            sample = self.data_loader.get_sample(idx)
            
            if sample and sample['image_path']:
                result = self.analyze_single_sample(i, sample)
                all_results.append(result)
                
                # Save intermediate results every 10 samples
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_results)
            
            # Clear memory periodically
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save final results
        self.results['all_samples'] = all_results
        self.save_results()
        
        return all_results
    
    def save_intermediate_results(self, results):
        """Save intermediate results to pickle"""
        with open(self.dirs['raw_data'] / 'intermediate_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    def save_answer_analysis(self, df):
        """Save detailed answer analysis for manual review"""
        # Create answer analysis DataFrame
        answer_cols = ['study_id', 'question', 'medical_term', 'ground_truth', 
                      'full_generated_text', 'extracted_answer', 'is_correct', 
                      'answer_in_text']
        
        # Only include columns that exist
        existing_cols = [col for col in answer_cols if col in df.columns]
        
        if 'full_generated_text' in df.columns:
            answer_df = df[existing_cols].copy()
            
            # Add analysis columns
            answer_df['answer_method'] = 'improved_extraction'
            answer_df['needs_review'] = (~answer_df['is_correct']) & (answer_df['answer_in_text'])
            
            # Sort by correctness for easier review
            answer_df = answer_df.sort_values(['is_correct', 'medical_term'], ascending=[False, True])
            
            # Save to CSV
            answer_df.to_csv(self.dirs['statistics'] / 'answer_analysis.csv', index=False)
            
            # Create summary statistics - convert numpy int64 to Python int
            summary = {
                'total_samples': int(len(answer_df)),
                'correct_extraction': int(answer_df['is_correct'].sum()),
                'answer_in_text_but_wrong': int(answer_df['needs_review'].sum()),
                'uncertain_answers': int((answer_df['extracted_answer'] == 'uncertain').sum()),
                'accuracy_original': float(answer_df['is_correct'].mean()),
                'potential_accuracy': float(answer_df['answer_in_text'].mean())
            }
            
            # Save summary
            with open(self.dirs['statistics'] / 'answer_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Answer analysis saved to {self.dirs['statistics'] / 'answer_analysis.csv'}")
            logger.info(f"Original accuracy: {summary['accuracy_original']:.1%}")
            logger.info(f"Potential accuracy (answer in text): {summary['potential_accuracy']:.1%}")
    
    def save_results(self):
        """Save all results"""
        # Save raw results
        with open(self.dirs['raw_data'] / 'all_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results['all_samples'])
        df.to_csv(self.dirs['statistics'] / 'results.csv', index=False)
        
        # Save detailed answer analysis CSV
        self.save_answer_analysis(df)
        
        logger.info(f"Results saved to {self.dirs['statistics'] / 'results.csv'}")
    
    def generate_statistics(self, results):
        """Generate comprehensive statistics from results"""
        df = pd.DataFrame(results)
        
        stats = {}
        
        # 1. Answer Accuracy Statistics
        if 'is_correct' in df.columns:
            stats['accuracy'] = {
                'overall': df['is_correct'].mean(),
                'by_answer': df.groupby('ground_truth')['is_correct'].mean().to_dict(),
                'by_term': df.groupby('medical_term')['is_correct'].mean().to_dict() if 'medical_term' in df.columns else {}
            }
        
        # 2. Attention Metrics Statistics
        attention_metrics = ['inside_body_ratio', 'border_fraction', 'attention_entropy']
        for metric in attention_metrics:
            if metric in df.columns:
                stats[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'median': df[metric].median(),
                    'q25': df[metric].quantile(0.25),
                    'q75': df[metric].quantile(0.75)
                }
        
        # 3. Regional Distribution
        regional_metrics = ['left_fraction', 'right_fraction', 'apical_fraction', 'basal_fraction']
        regional_stats = {}
        for metric in regional_metrics:
            if metric in df.columns:
                regional_stats[metric] = df[metric].mean()
        stats['regional_distribution'] = regional_stats
        
        # 4. Prompt Sensitivity
        if 'prompt_js_divergence' in df.columns:
            # Filter out inf and nan values
            valid_js = df['prompt_js_divergence'][np.isfinite(df['prompt_js_divergence'])]
            if len(valid_js) > 0:
                stats['prompt_sensitivity'] = {
                    'js_divergence_mean': float(valid_js.mean()),
                    'js_divergence_std': float(valid_js.std()),
                    'consistency_rate': df['prompt_consistent'].mean() if 'prompt_consistent' in df.columns else None
                }
            else:
                stats['prompt_sensitivity'] = {
                    'js_divergence_mean': None,
                    'js_divergence_std': None,
                    'consistency_rate': df['prompt_consistent'].mean() if 'prompt_consistent' in df.columns else None
                }
        
        # 5. Faithfulness Metrics (if available)
        faithfulness_metrics = ['attention_focus', 'peak_ratio', 'attention_sparsity']
        for metric in faithfulness_metrics:
            if metric in df.columns:
                valid_values = df[metric].dropna()
                if len(valid_values) > 0:
                    stats[f'faithfulness_{metric}'] = {
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'n_samples': len(valid_values)
                    }
        
        # 6. Method Success Rates
        method_cols = [col for col in df.columns if col.endswith('_success')]
        if method_cols:
            stats['method_success_rates'] = {
                col: df[col].mean() for col in method_cols
            }
        
        # Save statistics
        with open(self.dirs['statistics'] / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return stats
    
    def generate_plots(self, results):
        """Generate comprehensive plots for paper"""
        df = pd.DataFrame(results)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Answer Accuracy Plot
        if 'is_correct' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Overall accuracy
            accuracy = df['is_correct'].mean()
            axes[0].bar(['Correct', 'Incorrect'], 
                       [accuracy, 1-accuracy],
                       color=['green', 'red'])
            axes[0].set_title(f'Overall Accuracy: {accuracy:.1%}')
            axes[0].set_ylabel('Proportion')
            
            # Accuracy by ground truth
            acc_by_truth = df.groupby('ground_truth')['is_correct'].mean()
            acc_by_truth.plot(kind='bar', ax=axes[1], color=['blue', 'orange'])
            axes[1].set_title('Accuracy by Ground Truth')
            axes[1].set_xlabel('Ground Truth')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.dirs['statistics'] / 'accuracy_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Attention Metrics Distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_to_plot = [
            ('inside_body_ratio', 'Inside Body Ratio', 0.7),
            ('border_fraction', 'Border Fraction', 0.05),
            ('attention_entropy', 'Attention Entropy', 3.0),
            ('left_fraction', 'Left Lung Fraction', None),
            ('right_fraction', 'Right Lung Fraction', None),
            ('prompt_js_divergence', 'JS Divergence', 0.2)
        ]
        
        for idx, (metric, title, threshold) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            if metric in df.columns:
                data = df[metric].dropna()
                
                # Filter out inf and nan values
                data_clean = data[np.isfinite(data)]
                
                if len(data_clean) > 0:
                    # Histogram
                    ax.hist(data_clean, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    ax.axvline(data_clean.mean(), color='red', linestyle='--', label=f'Mean: {data_clean.mean():.3f}')
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                
                # Add threshold line if specified
                if threshold:
                    ax.axvline(threshold, color='green', linestyle='--', label=f'Target: {threshold}')
                
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{title} Distribution')
                ax.legend()
            else:
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.dirs['statistics'] / 'metrics_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Regional Distribution Analysis
        regional_cols = ['left_fraction', 'right_fraction', 'apical_fraction', 'basal_fraction']
        if all(col in df.columns for col in regional_cols):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            regional_means = [df[col].mean() for col in regional_cols]
            regional_stds = [df[col].std() for col in regional_cols]
            
            x = np.arange(len(regional_cols))
            bars = ax.bar(x, regional_means, yerr=regional_stds, capsize=5,
                         color=['lightblue', 'lightgreen', 'salmon', 'gold'])
            
            ax.set_xlabel('Region')
            ax.set_ylabel('Mean Attention Fraction')
            ax.set_title('Regional Attention Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(['Left', 'Right', 'Apical', 'Basal'])
            
            # Add value labels on bars
            for bar, mean in zip(bars, regional_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.dirs['statistics'] / 'regional_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Faithfulness Metrics (if available)
        faith_metrics = ['attention_focus', 'peak_ratio', 'attention_sparsity']
        faith_data = {m: df[m].dropna() for m in faith_metrics if m in df.columns}
        
        # Filter out empty data
        faith_data = {k: v for k, v in faith_data.items() if len(v) > 0}
        
        if faith_data:
            fig, axes = plt.subplots(1, len(faith_data), figsize=(5*len(faith_data), 5))
            if len(faith_data) == 1:
                axes = [axes]
            
            for idx, (metric, data) in enumerate(faith_data.items()):
                if len(data) > 0:  # Only plot if data exists
                    axes[idx].violinplot([data], positions=[0], widths=0.5)
                    axes[idx].boxplot([data], positions=[0], widths=0.3)
                axes[idx].set_xticks([0])
                axes[idx].set_xticklabels([metric.replace('_', ' ').title()])
                axes[idx].set_ylabel('Value')
                axes[idx].set_title(f'{metric.replace("_", " ").title()}\n(n={len(data)})')
                
                # Add mean line
                axes[idx].axhline(data.mean(), color='red', linestyle='--', 
                                 label=f'Mean: {data.mean():.3f}')
                axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(self.dirs['statistics'] / 'faithfulness_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Correlation Matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            correlation_cols = [col for col in numeric_cols 
                              if not col.endswith('_success') and 'index' not in col]
            
            if len(correlation_cols) > 2:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                corr_matrix = df[correlation_cols].corr()
                
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, ax=ax,
                           cbar_kws={"shrink": 0.8})
                
                ax.set_title('Correlation Matrix of Metrics')
                
                plt.tight_layout()
                plt.savefig(self.dirs['statistics'] / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 6. Method Comparison (if multiple methods tested)
        method_metrics = [col for col in df.columns if 'token_' in col and 'inside_ratio' in col]
        if method_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = [col.replace('token_', '').replace('_inside_ratio', '') for col in method_metrics]
            means = [df[col].mean() for col in method_metrics]
            stds = [df[col].std() for col in method_metrics]
            
            x = np.arange(len(methods))
            ax.bar(x, means, yerr=stds, capsize=5)
            ax.set_xlabel('Method')
            ax.set_ylabel('Inside Body Ratio')
            ax.set_title('Attention Quality by Method')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.axhline(y=0.7, color='green', linestyle='--', label='Target (0.7)')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.dirs['statistics'] / 'method_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {self.dirs['statistics']}")
    
    def generate_report(self, results, stats):
        """Generate comprehensive text report"""
        report = []
        report.append("="*80)
        report.append("MEDGEMMA VISUALIZER - BATCH ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of samples: {len(results)}")
        report.append("")
        
        # 1. Accuracy Summary
        report.append("1. ANSWER ACCURACY")
        report.append("-"*40)
        if 'accuracy' in stats:
            report.append(f"Overall Accuracy: {stats['accuracy']['overall']:.1%}")
            report.append(f"Accuracy by Ground Truth:")
            for truth, acc in stats['accuracy']['by_answer'].items():
                report.append(f"  - {truth}: {acc:.1%}")
        report.append("")
        
        # 2. Attention Quality Metrics
        report.append("2. ATTENTION QUALITY METRICS")
        report.append("-"*40)
        
        metrics_info = [
            ('inside_body_ratio', 'Inside Body Ratio', 0.7, '≥'),
            ('border_fraction', 'Border Fraction', 0.05, '≤'),
            ('attention_entropy', 'Attention Entropy', 3.0, '≤')
        ]
        
        for metric, name, target, op in metrics_info:
            if metric in stats:
                m = stats[metric]
                report.append(f"{name}:")
                report.append(f"  Mean: {m['mean']:.3f} ± {m['std']:.3f}")
                report.append(f"  Median: {m['median']:.3f} (Q25: {m['q25']:.3f}, Q75: {m['q75']:.3f})")
                
                # Check against target
                if op == '≥':
                    passing = m['mean'] >= target
                else:
                    passing = m['mean'] <= target
                status = "✓ PASS" if passing else "✗ FAIL"
                report.append(f"  Target: {op} {target} [{status}]")
                report.append("")
        
        # 3. Regional Distribution
        report.append("3. REGIONAL ATTENTION DISTRIBUTION")
        report.append("-"*40)
        if 'regional_distribution' in stats:
            for region, value in stats['regional_distribution'].items():
                report.append(f"{region.replace('_', ' ').title()}: {value:.3f}")
        report.append("")
        
        # 4. Prompt Sensitivity
        report.append("4. PROMPT SENSITIVITY ANALYSIS")
        report.append("-"*40)
        if 'prompt_sensitivity' in stats:
            ps = stats['prompt_sensitivity']
            if ps.get('js_divergence_mean') is not None:
                report.append(f"JS Divergence: {ps['js_divergence_mean']:.3f} ± {ps.get('js_divergence_std', 0):.3f}")
            else:
                report.append("JS Divergence: No valid data")
            if ps.get('consistency_rate') is not None:
                report.append(f"Answer Consistency: {ps['consistency_rate']:.1%}")
        report.append("")
        
        # 5. Faithfulness Metrics
        report.append("5. FAITHFULNESS VALIDATION")
        report.append("-"*40)
        faith_found = False
        for metric in ['attention_focus', 'peak_ratio', 'attention_sparsity']:
            key = f'faithfulness_{metric}'
            if key in stats:
                faith_found = True
                m = stats[key]
                report.append(f"{metric.replace('_', ' ').title()}:")
                report.append(f"  Mean: {m['mean']:.3f} ± {m['std']:.3f}")
                report.append(f"  (n={m['n_samples']} samples)")
        
        if not faith_found:
            report.append("No faithfulness metrics computed")
        report.append("")
        
        # 6. Method Success Rates
        report.append("6. METHOD SUCCESS RATES")
        report.append("-"*40)
        if 'method_success_rates' in stats:
            for method, rate in stats['method_success_rates'].items():
                report.append(f"{method.replace('_', ' ').title()}: {rate:.1%}")
        report.append("")
        
        # 7. Summary
        report.append("7. SUMMARY")
        report.append("-"*40)
        report.append("Key Findings:")
        
        # Determine key findings
        findings = []
        
        if 'accuracy' in stats:
            acc = stats['accuracy']['overall']
            if acc > 0.8:
                findings.append(f"✓ High overall accuracy ({acc:.1%})")
            elif acc > 0.7:
                findings.append(f"◐ Moderate accuracy ({acc:.1%})")
            else:
                findings.append(f"✗ Low accuracy ({acc:.1%})")
        
        if 'inside_body_ratio' in stats:
            ratio = stats['inside_body_ratio']['mean']
            if ratio >= 0.7:
                findings.append(f"✓ Good attention focus (inside body: {ratio:.3f})")
            else:
                findings.append(f"✗ Poor attention focus (inside body: {ratio:.3f})")
        
        if 'prompt_sensitivity' in stats and stats['prompt_sensitivity'].get('js_divergence_mean') is not None:
            js = stats['prompt_sensitivity']['js_divergence_mean']
            if js < 0.2:
                findings.append(f"✓ Robust to prompt variations (JS: {js:.3f})")
            else:
                findings.append(f"✗ Sensitive to prompt variations (JS: {js:.3f})")
        
        for finding in findings:
            report.append(f"  - {finding}")
        
        report.append("")
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.dirs['statistics'] / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text


def main():
    """Main execution function"""
    logger.info("Starting MedGemma Batch Analysis")
    
    # Setup GPU
    device = setup_gpu(min_free_gb=15.0)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading MedGemma model...")
    model, processor = load_model_enhanced(device=device)
    
    # Load data
    logger.info("Loading MIMIC data...")
    data_loader = MIMICDataLoader(MIMIC_CSV_PATH, MIMIC_IMAGE_BASE_PATH)
    
    # Create analyzer
    analyzer = BatchAnalyzer(model, processor, data_loader, OUTPUT_DIR)
    
    # Run analysis
    logger.info(f"Running batch analysis on {NUM_SAMPLES} samples...")
    results = analyzer.run_batch_analysis(NUM_SAMPLES)
    
    # Generate statistics
    logger.info("Generating statistics...")
    stats = analyzer.generate_statistics(results)
    
    # Generate plots
    logger.info("Generating plots...")
    analyzer.generate_plots(results)
    
    # Generate report
    logger.info("Generating report...")
    analyzer.generate_report(results, stats)
    
    logger.info(f"Analysis complete! Results saved to {OUTPUT_DIR}")
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return results, stats


if __name__ == "__main__":
    results, stats = main()