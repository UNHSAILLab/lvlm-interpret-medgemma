#!/usr/bin/env python3
"""
MedGemma-4b specific runner with proper architecture configuration
Adapted for MIMIC-CXR dataset with hard positive questions
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
from datetime import datetime

from medgemma_attention_extraction import SystemConfig, UnifiedAttentionExtractor
from medgemma_dataset_evaluator import MainEvaluator


class MedGemma4BEvaluator(MainEvaluator):
    """Extended evaluator for MedGemma-4b with Gemma3 architecture"""
    
    def __init__(self, model, processor, config: SystemConfig):
        super().__init__(model, processor, config)
        
        # Update architecture info based on Gemma3 model
        self.update_gemma3_architecture()
        
    def update_gemma3_architecture(self):
        """Update architecture parameters for Gemma3 model"""
        # From the model structure:
        # self_attn has q_proj: 2560->2048, k/v_proj: 2560->1024
        # This means H (from q) and K (from k/v) need to be derived
        
        # Calculate number of heads from dimensions
        # Assuming head_dim = 256 (common for Gemma)
        head_dim = 256
        q_features = 2048
        kv_features = 1024
        
        self.config.architecture.update({
            "H": q_features // head_dim,  # 2048/256 = 8 attention heads
            "K": kv_features // head_dim,  # 1024/256 = 4 key-value heads
            "mm_tokens": 256,  # After 4x4 avg pooling: (64x64)/16 = 256
            "grid_size": 16,
            "vision_patches": 64,  # 896/14 = 64 patches per side
            "pooling_factor": 4,  # AvgPool2d(4, 4)
            "text_hidden": 2560,
            "vision_hidden": 1152
        })
        
        self.logger.info(f"Updated Gemma3 architecture: {self.config.architecture}")
    
    def process_single_sample(
        self,
        row: pd.Series,
        image_root: Path,
        target_terms: List[str]
    ) -> Dict[str, Any]:
        """Process a single sample with Gemma3-specific handling"""
        
        # Handle MIMIC-CXR specific path
        image_path = image_root / row.image_path
        if not image_path.exists():
            # Try without extension changes
            image_path = image_root / row.dicom_id
            if not image_path.exists():
                image_path = image_root / f"{row.dicom_id}.jpg"
                if not image_path.exists():
                    self.logger.warning(f"Image not found: {row.image_path}")
                    return None
        
        image = Image.open(image_path).convert("RGB")
        question = row.question
        ground_truth = row.answer
        
        # For hard positive analysis, track variant info
        variant_info = {
            "baseline_question": row.get("baseline_question", question),
            "question_variant": row.get("question_variant", "original"),
            "variant_index": row.get("variant_index", 0),
            "strategy": row.get("strategy", "baseline")
        }
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Process with processor - handle MedGemma's specific format
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate with attention capture
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": self.config.decoding["max_new_tokens"],
                "do_sample": self.config.decoding["do_sample"],
                "output_attentions": True,
                "return_dict_in_generate": True
            }
            
            # Only add temperature if sampling is enabled
            if self.config.decoding["do_sample"]:
                gen_kwargs["temperature"] = self.config.decoding["temperature"]
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Extract answer
        answer_ids = outputs.sequences[0][prompt_len:]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True)
        
        # Re-run for consistent attention shapes
        full_ids = outputs.sequences[:, :-1]
        with torch.no_grad():
            attn_outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.get('pixel_values'),
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )
        
        # Token gating
        prompt_ids = full_ids[0, :prompt_len].tolist()
        gating_indices = self.extractor.token_gating.find_target_indices(
            self.processor.tokenizer,
            prompt_ids,
            question,
            target_terms
        )
        
        # Extract attention with Gemma3 awareness
        attention_result = self.extract_gemma3_attention(
            attn_outputs, prompt_len, full_ids, gating_indices, messages
        )
        
        # Body mask and metrics
        from unified_attention_system import BodyMaskGenerator, AttentionMetrics
        mask = BodyMaskGenerator.create_body_mask(image)
        masked_attention = BodyMaskGenerator.apply_body_mask(
            attention_result.grid, mask
        )
        metrics = AttentionMetrics.compute_metrics(attention_result.grid, mask)
        
        # Check correctness
        is_correct = self.normalize_answer(answer) == self.normalize_answer(ground_truth)
        
        return {
            "dicom_id": row.dicom_id,
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
            **variant_info,
            **attention_result.metadata
        }
    
    def extract_gemma3_attention(
        self,
        attn_outputs,
        prompt_len: int,
        full_ids: torch.Tensor,
        gating_indices: List[int],
        messages: List
    ):
        """Extract attention with Gemma3-specific handling"""
        
        # The model has self-attention in language_model.layers
        # No explicit cross-attention, so we'll use self-attention
        
        # Update extractor with correct architecture
        self.extractor.config.architecture = self.config.architecture
        
        # Try extraction with self-attention priority for Gemma3
        return self.extractor.extract_attention_maps(
            attn_outputs,
            prompt_len,
            full_ids,
            gating_indices,
            mode_order=["self", "gradcam", "uniform"],  # No cross-attention in Gemma3
            messages=messages
        )
    
    def analyze_hard_positives(self, results_df: pd.DataFrame):
        """Analyze performance on hard positive question variants"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("HARD POSITIVE ANALYSIS")
        self.logger.info("="*60)
        
        # Group by baseline question
        if "baseline_question" in results_df.columns:
            grouped = results_df.groupby("baseline_question")
            
            consistency_scores = []
            for baseline_q, group in grouped:
                if len(group) > 1:
                    # Check answer consistency
                    answers = group["answer"].values
                    unique_answers = len(set(answers))
                    consistency = 1.0 if unique_answers == 1 else 0.0
                    consistency_scores.append(consistency)
                    
                    # Check attention consistency (JS divergence)
                    if len(group) >= 2:
                        from unified_attention_system import AttentionMetrics
                        grids = [row["attention_grid"] for _, row in group.iterrows()]
                        js_divs = []
                        for i in range(len(grids)-1):
                            for j in range(i+1, len(grids)):
                                js_div = AttentionMetrics.map_similarity(grids[i], grids[j])
                                js_divs.append(js_div)
                        
                        avg_js = np.mean(js_divs) if js_divs else 0
                        
                        self.logger.info(f"\nBaseline: {baseline_q[:50]}...")
                        self.logger.info(f"  Variants: {len(group)}")
                        self.logger.info(f"  Answer consistency: {consistency}")
                        self.logger.info(f"  Avg JS divergence: {avg_js:.3f}")
                        self.logger.info(f"  Strategies: {group['strategy'].unique().tolist()}")
            
            if consistency_scores:
                self.logger.info(f"\nOverall answer consistency: {np.mean(consistency_scores):.3f}")
        
        # Analyze by strategy
        if "strategy" in results_df.columns:
            self.logger.info("\nAccuracy by strategy:")
            for strategy in results_df["strategy"].unique():
                strategy_df = results_df[results_df["strategy"] == strategy]
                acc = strategy_df["correct"].mean()
                self.logger.info(f"  {strategy}: {acc:.3f} (n={len(strategy_df)})")


def load_medgemma4b(device: str = "cuda"):
    """Load MedGemma-4b model with proper configuration"""
    
    model_id = "google/medgemma-4b-it"
    
    print(f"Loading MedGemma-4b model...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    
    # Load model with eager attention for attention outputs
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Required for output_attentions
    )
    
    # Enable attention outputs (now safe with eager implementation)
    model.config.output_attentions = True
    model.config.return_dict = True
    
    # Move to device if not using device_map
    if device != "cuda":
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Vision tower: SigLIP with 27 layers")
    print(f"  Language model: Gemma3 with 34 layers")
    print(f"  Image tokens: 64x64 patches -> 16x16 after pooling = 256 tokens")
    
    return model, processor


def main():
    """Main execution for MedGemma-4b evaluation"""
    
    # Configuration for MIMIC-CXR
    config = SystemConfig(
        model_id="google/medgemma-4b-it",
        device="cuda" if torch.cuda.is_available() else "cpu",
        io_paths={
            "data_csv": Path("/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv"),
            "image_root": Path("/home/bsada1/mimic_cxr_hundred_vqa"),
            "out_dir": Path("./medgemma4b_results")
        },
        eval={
            "target_terms": [
                "effusion", "pneumothorax", "consolidation", "pneumonia",
                "edema", "cardiomegaly", "atelectasis", "fracture",
                "opacity", "infiltrate", "nodule", "mass"
            ],
            "robustness": {
                "paraphrases": [],  # Already in dataset as hard positives
                "perturbations": [
                    {"type": "gaussian_noise", "params": {"sigma": 2}},
                    {"type": "contrast", "params": {"factor": 1.2}},
                    {"type": "one_pixel", "params": {"epsilon": 0.1, "num_pixels": 5}}
                ]
            }
        },
        decoding={
            "max_new_tokens": 100,
            "temperature": 0.0,
            "do_sample": False
        }
    )
    
    print("="*60)
    print("MedGemma-4b Evaluation on MIMIC-CXR")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {config.model_id}")
    print(f"  Device: {config.device}")
    print(f"  Dataset: MIMIC-CXR with hard positives")
    print(f"  Output: {config.io_paths['out_dir']}")
    print("="*60)
    
    # Load model
    model, processor = load_medgemma4b(config.device)
    
    # Create evaluator
    evaluator = MedGemma4BEvaluator(model, processor, config)
    
    # Run evaluation
    print("\nStarting evaluation...")
    
    # Load and check dataset
    df = pd.read_csv(config.io_paths["data_csv"])
    print(f"Loaded {len(df)} samples")
    print(f"Unique baseline questions: {df['baseline_question'].nunique()}")
    print(f"Question strategies: {df['strategy'].unique()}")
    
    # Evaluate with limit for testing
    results_df = evaluator.evaluate_dataset(
        sample_limit=50,  # Start with 50 samples for testing
        run_robustness=False  # Skip robustness for hard positive analysis
    )
    
    # Analyze hard positives
    evaluator.analyze_hard_positives(results_df)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {config.io_paths['out_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()