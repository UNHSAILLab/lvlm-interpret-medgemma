#!/usr/bin/env python3
"""
Main evaluation runner for medical VQA with attention analysis
Processes datasets, extracts attention, computes metrics, and saves results
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import csv
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

from medgemma_attention_extraction import (
    SystemConfig,
    UnifiedAttentionExtractor,
    TokenGating,
    BodyMaskGenerator,
    AttentionMetrics,
    Visualizer
)
from medgemma_robustness_eval import (
    RobustnessEvaluator,
    ImagePerturbations
)


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class MainEvaluator:
    """Main evaluation pipeline for medical VQA"""
    
    def __init__(self, model, processor, config: SystemConfig):
        self.model = model
        self.processor = processor
        self.config = config
        self.extractor = UnifiedAttentionExtractor(model, processor, config)
        self.robustness_evaluator = RobustnessEvaluator(
            model, processor, self.extractor, config
        )
        
        # Setup output directories
        self.setup_directories()
        self.logger = setup_logging(self.config.io_paths["out_dir"])
        
    def setup_directories(self):
        """Create output directory structure"""
        base_dir = self.config.io_paths["out_dir"]
        
        self.dirs = {
            "base": base_dir,
            "overlays": base_dir / "overlays",
            "grids": base_dir / "grids",
            "visualizations": base_dir / "visualizations",
            "robustness": base_dir / "robustness",
            "analysis": base_dir / "analysis"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, csv_path: Path) -> pd.DataFrame:
        """Load dataset from CSV"""
        self.logger.info(f"Loading dataset from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_cols = ["study_id", "question", "correct_answer", "image_path"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try alternative column names
            col_mapping = {
                "study_id": ["StudyID", "study", "id"],
                "question": ["Question", "query", "text"],
                "correct_answer": ["answer", "ground_truth", "label"],
                "image_path": ["ImagePath", "image", "path"]
            }
            
            for required, alternatives in col_mapping.items():
                if required in missing_cols:
                    for alt in alternatives:
                        if alt in df.columns:
                            df[required] = df[alt]
                            missing_cols.remove(required)
                            break
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} samples")
        return df
    
    def process_single_sample(
        self,
        row: pd.Series,
        image_root: Path,
        target_terms: List[str]
    ) -> Dict[str, Any]:
        """Process a single sample with attention extraction"""
        
        # Load image
        image_path = image_root / row.image_path
        if not image_path.exists():
            self.logger.warning(f"Image not found: {image_path}")
            return None
        
        image = Image.open(image_path).convert("RGB")
        question = row.question
        ground_truth = row.correct_answer
        
        # Create messages for model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate with attention
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.config.decoding["max_new_tokens"],
                "do_sample": self.config.decoding["do_sample"],
                "temperature": self.config.decoding["temperature"],
                "output_attentions": True,
                "return_dict_in_generate": True
            }
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Extract answer
        answer_ids = outputs.sequences[0][prompt_len:]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True)
        
        # Re-run forward pass for consistent attention
        full_ids = outputs.sequences[:, :-1]
        with torch.no_grad():
            attn_outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.get('pixel_values'),
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )
        
        # Extract gating indices
        prompt_ids = full_ids[0, :prompt_len].tolist()
        gating_indices = TokenGating.find_target_indices(
            self.processor.tokenizer,
            prompt_ids,
            question,
            target_terms
        )
        
        # Extract attention maps
        attention_result = self.extractor.extract_attention_maps(
            attn_outputs,
            prompt_len,
            full_ids,
            gating_indices,
            mode_order=["cross", "self", "gradcam", "uniform"],
            messages=messages
        )
        
        # Generate body mask
        mask = BodyMaskGenerator.create_body_mask(image)
        
        # Apply mask to attention
        masked_attention = BodyMaskGenerator.apply_body_mask(
            attention_result.grid,
            mask
        )
        
        # Compute metrics
        metrics = AttentionMetrics.compute_metrics(attention_result.grid, mask)
        
        # Check correctness
        is_correct = self.normalize_answer(answer) == self.normalize_answer(ground_truth)
        
        return {
            "study_id": row.study_id,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "attention_mode": attention_result.mode,
            "gate_count": len(gating_indices),
            "attention_grid": attention_result.grid,
            "masked_attention": masked_attention,
            "body_mask": mask,
            "image": image,
            **metrics,
            **attention_result.metadata
        }
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.lower().strip()
        
        # Handle yes/no variations
        if any(word in answer for word in ["yes", "positive", "present", "detected"]):
            return "yes"
        elif any(word in answer for word in ["no", "negative", "absent", "not detected"]):
            return "no"
        
        return answer
    
    def save_sample_outputs(self, result: Dict[str, Any]):
        """Save outputs for a single sample"""
        study_id = result["study_id"]
        
        # Save attention grid
        np.save(
            self.dirs["grids"] / f"{study_id}_grid.npy",
            result["attention_grid"]
        )
        
        # Save overlay visualization
        overlay = Visualizer.create_overlay(
            result["image"],
            result["masked_attention"]
        )
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(self.dirs["overlays"] / f"{study_id}_overlay.png")
        
        # Save comprehensive visualization
        Visualizer.save_visualization(
            result["image"],
            result["attention_grid"],
            result["body_mask"],
            self.dirs["visualizations"] / f"{study_id}_full.png",
            title=f"Study {study_id}: {result['attention_mode']} attention"
        )
        
        # Save metadata - convert numpy types to Python types for JSON serialization
        metadata = {}
        for k, v in result.items():
            if k not in ["attention_grid", "masked_attention", "body_mask", "image"]:
                # Convert numpy types to Python types
                if isinstance(v, (np.float32, np.float64)):
                    metadata[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    metadata[k] = int(v)
                elif isinstance(v, np.ndarray):
                    metadata[k] = v.tolist()
                elif isinstance(v, (bool, np.bool_)):
                    metadata[k] = bool(v)
                else:
                    metadata[k] = v
        
        with open(self.dirs["grids"] / f"{study_id}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def evaluate_dataset(
        self,
        csv_path: Optional[Path] = None,
        image_root: Optional[Path] = None,
        target_terms: Optional[List[str]] = None,
        run_robustness: bool = True,
        sample_limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Main evaluation pipeline"""
        
        # Use config defaults if not provided
        csv_path = csv_path or self.config.io_paths["data_csv"]
        image_root = image_root or self.config.io_paths["image_root"]
        target_terms = target_terms or self.config.eval["target_terms"]
        
        # Load dataset
        df = self.load_dataset(csv_path)
        
        if sample_limit:
            df = df.head(sample_limit)
            self.logger.info(f"Limited to {sample_limit} samples")
        
        # Process samples
        all_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            try:
                result = self.process_single_sample(row, image_root, target_terms)
                
                if result:
                    all_results.append(result)
                    self.save_sample_outputs(result)
                    
                    # Run robustness evaluation on subset
                    if run_robustness and idx < 10:  # Limit robustness to first 10
                        self.run_robustness_evaluation(result, target_terms)
                        
            except Exception as e:
                import traceback
                self.logger.error(f"Error processing {row.study_id}: {e}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save main results CSV
        csv_path = self.dirs["base"] / "evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved results to {csv_path}")
        
        # Generate analysis plots
        self.generate_analysis_plots(results_df)
        
        # Print summary statistics
        self.print_summary(results_df)
        
        return results_df
    
    def run_robustness_evaluation(
        self,
        result: Dict[str, Any],
        target_terms: List[str]
    ):
        """Run robustness evaluation for a sample"""
        
        study_id = result["study_id"]
        image = result["image"]
        question = result["question"]
        
        # Paraphrase evaluation
        paraphrases = self.config.eval["robustness"]["paraphrases"]
        paraphrase_results, paraphrase_comparisons = self.robustness_evaluator.evaluate_paraphrases(
            image, question, paraphrases, target_terms
        )
        
        # Perturbation evaluation
        perturbations = self.config.eval["robustness"]["perturbations"]
        perturbation_results = self.robustness_evaluator.evaluate_perturbations(
            image, question, perturbations, target_terms
        )
        
        # Save results
        self.robustness_evaluator.save_robustness_results(
            self.dirs["robustness"],
            study_id,
            paraphrase_results,
            paraphrase_comparisons,
            perturbation_results
        )
    
    def generate_analysis_plots(self, df: pd.DataFrame):
        """Generate analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Attention mode distribution
        mode_counts = df['attention_mode'].value_counts()
        axes[0, 0].bar(mode_counts.index, mode_counts.values)
        axes[0, 0].set_title('Attention Mode Distribution')
        axes[0, 0].set_xlabel('Mode')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Accuracy by attention mode
        accuracy_by_mode = df.groupby('attention_mode')['correct'].mean()
        axes[0, 1].bar(accuracy_by_mode.index, accuracy_by_mode.values)
        axes[0, 1].set_title('Accuracy by Attention Mode')
        axes[0, 1].set_xlabel('Mode')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Entropy distribution
        axes[0, 2].hist(df['entropy'], bins=30, edgecolor='black')
        axes[0, 2].set_title('Attention Entropy Distribution')
        axes[0, 2].set_xlabel('Entropy')
        axes[0, 2].set_ylabel('Count')
        
        # 4. Focus vs Inside Body Ratio
        axes[1, 0].scatter(df['focus'], df['inside_body_ratio'], alpha=0.5)
        axes[1, 0].set_title('Focus vs Inside Body Ratio')
        axes[1, 0].set_xlabel('Focus (Max Attention)')
        axes[1, 0].set_ylabel('Inside Body Ratio')
        
        # 5. Spatial distribution
        spatial_data = df[['left_fraction', 'right_fraction', 'apical_fraction', 'basal_fraction']].mean()
        axes[1, 1].bar(spatial_data.index, spatial_data.values)
        axes[1, 1].set_title('Average Spatial Distribution')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Fraction')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Gate count distribution
        axes[1, 2].hist(df['gate_count'], bins=20, edgecolor='black')
        axes[1, 2].set_title('Gate Count Distribution')
        axes[1, 2].set_xlabel('Number of Gated Tokens')
        axes[1, 2].set_ylabel('Count')
        
        plt.suptitle('Medical VQA Attention Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.dirs["analysis"] / "summary_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap
        numeric_cols = ['entropy', 'focus', 'inside_body_ratio', 'border_fraction',
                       'left_fraction', 'right_fraction', 'apical_fraction', 'basal_fraction']
        
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(self.dirs["analysis"] / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*60)
        
        # Overall statistics
        self.logger.info(f"Total samples: {len(df)}")
        self.logger.info(f"Overall accuracy: {df['correct'].mean():.3f}")
        
        # By attention mode
        self.logger.info("\nAccuracy by attention mode:")
        for mode in df['attention_mode'].unique():
            mode_df = df[df['attention_mode'] == mode]
            self.logger.info(f"  {mode}: {mode_df['correct'].mean():.3f} (n={len(mode_df)})")
        
        # Metric averages
        self.logger.info("\nAverage metrics:")
        metrics = ['entropy', 'focus', 'inside_body_ratio', 'border_fraction']
        for metric in metrics:
            self.logger.info(f"  {metric}: {df[metric].mean():.3f} ± {df[metric].std():.3f}")
        
        # Spatial distribution
        self.logger.info("\nAverage spatial distribution:")
        self.logger.info(f"  Left: {df['left_fraction'].mean():.3f}")
        self.logger.info(f"  Right: {df['right_fraction'].mean():.3f}")
        self.logger.info(f"  Apical: {df['apical_fraction'].mean():.3f}")
        self.logger.info(f"  Basal: {df['basal_fraction'].mean():.3f}")
        
        self.logger.info("="*60)


def main():
    """Main entry point"""
    
    # Example configuration
    config = SystemConfig(
        model_id="google/gemma-2b-vqa",
        io_paths={
            "data_csv": Path("./mimic_test_data.csv"),
            "image_root": Path("./mimic_images"),
            "out_dir": Path("./evaluation_results")
        },
        eval={
            "target_terms": ["effusion", "pneumothorax", "consolidation", "pneumonia", "edema"],
            "robustness": {
                "paraphrases": [
                    "Is there pleural effusion visible?",
                    "Can you see any fluid in the pleural space?",
                    "Does this X-ray show effusion?"
                ],
                "perturbations": [
                    {"type": "gaussian_noise", "params": {"sigma": 2}},
                    {"type": "contrast", "params": {"factor": 1.2}},
                    {"type": "brightness", "params": {"factor": 0.9}},
                    {"type": "one_pixel", "params": {"epsilon": 0.1, "num_pixels": 5}}
                ]
            }
        }
    )
    
    print("Medical VQA Evaluation System")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {config.model_id}")
    print(f"  Device: {config.device}")
    print(f"  Output directory: {config.io_paths['out_dir']}")
    print(f"  Target terms: {config.eval['target_terms']}")
    print("="*60)
    
    # Note: In actual use, load real model and processor
    # from transformers import AutoModelForImageTextToText, AutoProcessor
    # model = AutoModelForImageTextToText.from_pretrained(config.model_id)
    # processor = AutoProcessor.from_pretrained(config.model_id)
    # evaluator = MainEvaluator(model, processor, config)
    # results = evaluator.evaluate_dataset(sample_limit=100)
    
    print("\n✓ Evaluation system ready!")
    print("To run evaluation:")
    print("  1. Load your model and processor")
    print("  2. Create MainEvaluator instance")
    print("  3. Call evaluate_dataset()")


if __name__ == "__main__":
    main()