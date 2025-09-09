#!/usr/bin/env python3
"""
Robustness evaluation framework for medical VQA
Tests paraphrases and perturbations
"""

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from medgemma_attention_extraction import (
    UnifiedAttentionExtractor,
    TokenGating,
    BodyMaskGenerator,
    AttentionMetrics,
    AttentionOutput,
    SystemConfig
)


@dataclass
class RobustnessResult:
    """Container for robustness test results"""
    question: str
    answer: str
    attention_grid: np.ndarray
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImagePerturbations:
    """Apply various perturbations to test robustness"""
    
    @staticmethod
    def apply_perturbation(
        image: Image.Image,
        perturbation: Dict[str, Any]
    ) -> Image.Image:
        """Apply specified perturbation to image"""
        
        perturb_type = perturbation["type"]
        params = perturbation.get("params", {})
        
        if perturb_type == "gaussian_noise":
            return ImagePerturbations.add_gaussian_noise(image, **params)
        elif perturb_type == "contrast":
            return ImagePerturbations.adjust_contrast(image, **params)
        elif perturb_type == "brightness":
            return ImagePerturbations.adjust_brightness(image, **params)
        elif perturb_type == "blur":
            return ImagePerturbations.apply_blur(image, **params)
        elif perturb_type == "one_pixel":
            return ImagePerturbations.one_pixel_attack(image, **params)
        elif perturb_type == "rotation":
            return ImagePerturbations.rotate_image(image, **params)
        elif perturb_type == "preprocess_variant":
            return ImagePerturbations.preprocessing_variant(image, **params)
        else:
            return image
    
    @staticmethod
    def add_gaussian_noise(
        image: Image.Image,
        sigma: float = 2.0
    ) -> Image.Image:
        """Add Gaussian noise to image"""
        
        img_array = np.array(image)
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    @staticmethod
    def adjust_contrast(
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """Adjust image contrast"""
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_brightness(
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """Adjust image brightness"""
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_blur(
        image: Image.Image,
        radius: float = 1.0
    ) -> Image.Image:
        """Apply Gaussian blur"""
        
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def one_pixel_attack(
        image: Image.Image,
        epsilon: float = 0.1,
        num_pixels: int = 5
    ) -> Image.Image:
        """Apply bounded one-pixel perturbation"""
        
        img_array = np.array(image).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create body mask to constrain perturbations
        mask = BodyMaskGenerator.create_body_mask(image)
        
        # Find valid pixel locations within body
        valid_y, valid_x = np.where(mask > 0)
        
        if len(valid_y) > 0:
            # Random selection of pixels
            num_pixels = min(num_pixels, len(valid_y))
            indices = np.random.choice(len(valid_y), num_pixels, replace=False)
            
            for idx in indices:
                y, x = valid_y[idx], valid_x[idx]
                
                # Bounded perturbation
                max_val = 255
                current_val = img_array[y, x]
                
                # Calculate bounded change
                change = np.random.uniform(-epsilon * max_val, epsilon * max_val, 
                                          current_val.shape)
                
                # Apply and clip
                if len(img_array.shape) == 3:
                    img_array[y, x] = np.clip(current_val + change, 0, 255)
                else:
                    img_array[y, x] = np.clip(current_val + change, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    @staticmethod
    def rotate_image(
        image: Image.Image,
        angle: float = 5.0
    ) -> Image.Image:
        """Rotate image by small angle"""
        
        return image.rotate(angle, fillcolor='black')
    
    @staticmethod
    def preprocessing_variant(
        image: Image.Image,
        resize_mode: str = "center_crop"
    ) -> Image.Image:
        """Apply different preprocessing strategies"""
        
        if resize_mode == "center_crop":
            # Center crop to square
            w, h = image.size
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            return image.crop((left, top, left + size, top + size))
            
        elif resize_mode == "pad":
            # Pad to square
            w, h = image.size
            size = max(w, h)
            new_img = Image.new('RGB', (size, size), 'black')
            paste_x = (size - w) // 2
            paste_y = (size - h) // 2
            new_img.paste(image, (paste_x, paste_y))
            return new_img
            
        else:
            return image


class RobustnessEvaluator:
    """Evaluate model robustness to paraphrases and perturbations"""
    
    def __init__(
        self,
        model,
        processor,
        extractor: UnifiedAttentionExtractor,
        config: SystemConfig
    ):
        self.model = model
        self.processor = processor
        self.extractor = extractor
        self.config = config
        
    def evaluate_paraphrases(
        self,
        image: Image.Image,
        base_question: str,
        paraphrases: List[str],
        target_terms: List[str]
    ) -> Tuple[List[RobustnessResult], List[Dict[str, Any]]]:
        """Evaluate robustness to question paraphrases"""
        
        all_questions = [base_question] + paraphrases
        results = []
        
        for question in all_questions:
            result = self.run_single_evaluation(image, question, target_terms)
            results.append(result)
        
        # Compute pairwise comparisons
        comparisons = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                js_div = AttentionMetrics.map_similarity(
                    results[i].attention_grid,
                    results[j].attention_grid
                )
                
                answer_flip = results[i].answer != results[j].answer
                
                comparisons.append({
                    "question_i": all_questions[i],
                    "question_j": all_questions[j],
                    "js_divergence": js_div,
                    "answer_flip": answer_flip,
                    "answer_i": results[i].answer,
                    "answer_j": results[j].answer
                })
        
        return results, comparisons
    
    def evaluate_perturbations(
        self,
        image: Image.Image,
        question: str,
        perturbations: List[Dict[str, Any]],
        target_terms: List[str]
    ) -> List[RobustnessResult]:
        """Evaluate robustness to image perturbations"""
        
        results = []
        
        # Original image
        original_result = self.run_single_evaluation(image, question, target_terms)
        original_result.metadata["perturbation"] = "original"
        results.append(original_result)
        
        # Apply each perturbation
        for perturbation in perturbations:
            perturbed_image = ImagePerturbations.apply_perturbation(image, perturbation)
            result = self.run_single_evaluation(perturbed_image, question, target_terms)
            result.metadata["perturbation"] = perturbation
            results.append(result)
        
        return results
    
    def run_single_evaluation(
        self,
        image: Image.Image,
        question: str,
        target_terms: List[str]
    ) -> RobustnessResult:
        """Run single evaluation with attention extraction"""
        
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
        
        # Re-run forward pass for consistent attention shapes
        full_ids = outputs.sequences[:, :-1]
        with torch.no_grad():
            attn_outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.get('pixel_values'),
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )
        
        # Extract prompt token indices for gating
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
        
        # Generate body mask and compute metrics
        mask = BodyMaskGenerator.create_body_mask(image)
        metrics = AttentionMetrics.compute_metrics(attention_result.grid, mask)
        
        return RobustnessResult(
            question=question,
            answer=answer,
            attention_grid=attention_result.grid,
            metrics=metrics,
            metadata={
                "attention_mode": attention_result.mode,
                "gate_count": len(gating_indices),
                **attention_result.metadata
            }
        )
    
    def save_robustness_results(
        self,
        output_dir: Path,
        study_id: str,
        paraphrase_results: List[RobustnessResult],
        paraphrase_comparisons: List[Dict],
        perturbation_results: List[RobustnessResult]
    ):
        """Save robustness evaluation results"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save paraphrase results
        paraphrase_data = {
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "metrics": r.metrics,
                    "metadata": r.metadata
                }
                for r in paraphrase_results
            ],
            "comparisons": paraphrase_comparisons
        }
        
        with open(output_dir / f"{study_id}_paraphrase_robustness.json", "w") as f:
            json.dump(paraphrase_data, f, indent=2)
        
        # Save perturbation results
        perturbation_data = {
            "results": [
                {
                    "perturbation": r.metadata.get("perturbation", "unknown"),
                    "answer": r.answer,
                    "metrics": r.metrics,
                    "metadata": r.metadata
                }
                for r in perturbation_results
            ]
        }
        
        with open(output_dir / f"{study_id}_perturbation_robustness.json", "w") as f:
            json.dump(perturbation_data, f, indent=2)
        
        # Save attention grids
        np.savez_compressed(
            output_dir / f"{study_id}_attention_grids.npz",
            paraphrase_grids=np.array([r.attention_grid for r in paraphrase_results]),
            perturbation_grids=np.array([r.attention_grid for r in perturbation_results])
        )


def test_robustness():
    """Test robustness evaluation"""
    print("Testing Robustness Evaluator...")
    
    # Test perturbations
    test_image = Image.new('RGB', (512, 512), 'white')
    
    perturbations = [
        {"type": "gaussian_noise", "params": {"sigma": 2}},
        {"type": "contrast", "params": {"factor": 1.2}},
        {"type": "one_pixel", "params": {"epsilon": 0.1}}
    ]
    
    for perturb in perturbations:
        result = ImagePerturbations.apply_perturbation(test_image, perturb)
        print(f"Applied {perturb['type']}: {result.size}")
    
    print("✓ Robustness tests passed!")


if __name__ == "__main__":
    test_robustness()