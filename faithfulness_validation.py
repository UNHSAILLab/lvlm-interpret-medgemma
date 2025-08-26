#!/usr/bin/env python3
"""
Faithfulness Validation Module for MedGemma Attention Visualization
Implements deletion/insertion curves and comprehensiveness/sufficiency metrics
As recommended by advisor feedback
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm

class FaithfulnessValidator:
    """Quantitative validation of attention map faithfulness"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        
    def get_target_logprob(self, image, prompt, target_word, return_all=False):
        """Get log probability of target word during generation"""
        # Prepare inputs
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
        
        # Generate and get logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Find target word in generated sequence
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Get tokens for target word
        target_tokens = self.processor.tokenizer.encode(target_word, add_special_tokens=False)
        
        # Find target tokens in generated sequence
        target_logprob = 0.0
        found = False
        
        for i in range(len(generated_ids) - len(target_tokens) + 1):
            if all(generated_ids[i+j] == target_tokens[j] for j in range(len(target_tokens))):
                # Found target word - compute log prob
                for j, token_id in enumerate(target_tokens):
                    if i+j < len(outputs.scores):
                        logits = outputs.scores[i+j][0]
                        probs = F.softmax(logits, dim=-1)
                        target_logprob += torch.log(probs[token_id]).item()
                found = True
                break
        
        if not found:
            # Target word not generated - return very low log prob
            target_logprob = -100.0
        
        if return_all:
            return target_logprob, generated_text
        return target_logprob
    
    def mask_patches(self, image, attention_map, percentile, mask_type='delete'):
        """Mask top-k percentile of patches based on attention"""
        # Ensure attention map matches image resolution
        H, W = image.shape[:2] if len(image.shape) == 2 else image.shape[1:3]
        
        # Resize attention to image size
        if attention_map.shape != (H, W):
            attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            attention_resized = attention_map
        
        # Get threshold for top-k percentile
        threshold = np.percentile(attention_resized.flatten(), 100 - percentile)
        mask = attention_resized >= threshold
        
        # Apply mask
        masked_image = image.copy()
        if mask_type == 'delete':
            # Replace with gray
            if len(image.shape) == 3:
                masked_image[mask] = [128, 128, 128]
            else:
                masked_image[mask] = 128
        elif mask_type == 'blur':
            # Apply Gaussian blur to masked regions
            blurred = gaussian_filter(image, sigma=5)
            masked_image[mask] = blurred[mask]
        
        return masked_image
    
    def deletion_curve(self, image, attention_map, prompt, target_word, 
                      percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
        """
        Compute deletion curve: progressively delete important patches
        and measure drop in target word probability
        """
        # Get baseline log probability
        baseline_logprob, baseline_text = self.get_target_logprob(
            image, prompt, target_word, return_all=True
        )
        
        print(f"Baseline generation: {baseline_text[:100]}")
        print(f"Baseline log prob for '{target_word}': {baseline_logprob:.3f}")
        
        curve = []
        for p in percentiles:
            # Mask top p% of patches
            masked_image = self.mask_patches(image, attention_map, p, 'delete')
            
            # Get new log probability
            new_logprob = self.get_target_logprob(masked_image, prompt, target_word)
            
            # Compute drop
            drop = baseline_logprob - new_logprob
            curve.append((p, drop))
            
            print(f"Deleted top {p}%: log prob = {new_logprob:.3f}, drop = {drop:.3f}")
        
        return curve
    
    def insertion_curve(self, image, attention_map, prompt, target_word,
                       percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
        """
        Compute insertion curve: start from blurred image and progressively
        reveal important patches
        """
        # Create fully blurred baseline
        blurred_baseline = gaussian_filter(image, sigma=10)
        
        curve = []
        for p in percentiles:
            # Create image with top p% revealed
            H, W = image.shape[:2] if len(image.shape) == 2 else image.shape[1:3]
            
            if attention_map.shape != (H, W):
                attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
            else:
                attention_resized = attention_map
            
            threshold = np.percentile(attention_resized.flatten(), 100 - p)
            reveal_mask = attention_resized >= threshold
            
            revealed_image = blurred_baseline.copy()
            revealed_image[reveal_mask] = image[reveal_mask]
            
            # Get log probability
            logprob = self.get_target_logprob(revealed_image, prompt, target_word)
            curve.append((p, logprob))
            
            print(f"Revealed top {p}%: log prob = {logprob:.3f}")
        
        return curve
    
    def compute_auc(self, curve):
        """Compute area under curve"""
        if not curve:
            return 0.0
        
        # Sort by x-axis (percentile)
        curve_sorted = sorted(curve, key=lambda x: x[0])
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(curve_sorted)):
            x1, y1 = curve_sorted[i-1]
            x2, y2 = curve_sorted[i]
            auc += 0.5 * (y1 + y2) * (x2 - x1) / 100  # Normalize by 100
        
        return auc
    
    def comprehensiveness_sufficiency(self, image, attention_map, prompt, target_word, 
                                     top_k=20):
        """
        Compute comprehensiveness and sufficiency metrics
        
        Comprehensiveness: How much does removing important regions hurt?
        Sufficiency: How well do important regions alone perform?
        """
        # Get baseline
        baseline_logprob = self.get_target_logprob(image, prompt, target_word)
        
        # Comprehensiveness: Remove top k%
        masked_image = self.mask_patches(image, attention_map, top_k, 'delete')
        masked_logprob = self.get_target_logprob(masked_image, prompt, target_word)
        comprehensiveness = (baseline_logprob - masked_logprob) / abs(baseline_logprob + 1e-8)
        
        # Sufficiency: Keep only top k%
        H, W = image.shape[:2] if len(image.shape) == 2 else image.shape[1:3]
        
        if attention_map.shape != (H, W):
            attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            attention_resized = attention_map
        
        threshold = np.percentile(attention_resized.flatten(), 100 - top_k)
        keep_mask = attention_resized >= threshold
        
        # Create image with only important regions
        kept_image = np.ones_like(image) * 128  # Gray background
        kept_image[keep_mask] = image[keep_mask]
        
        kept_logprob = self.get_target_logprob(kept_image, prompt, target_word)
        sufficiency = kept_logprob / (baseline_logprob + 1e-8)
        
        return comprehensiveness, sufficiency
    
    def visualize_curves(self, deletion_curve, insertion_curve, title="Faithfulness Curves"):
        """Visualize deletion and insertion curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Deletion curve
        if deletion_curve:
            x, y = zip(*deletion_curve)
            ax1.plot(x, y, 'b-', linewidth=2, marker='o')
            ax1.fill_between(x, 0, y, alpha=0.3)
            ax1.set_xlabel('Percentage of Patches Deleted')
            ax1.set_ylabel('Drop in Log Probability')
            ax1.set_title('Deletion Curve')
            ax1.grid(True, alpha=0.3)
            
            # Add AUC
            auc = self.compute_auc(deletion_curve)
            ax1.text(0.1, 0.9, f'AUC: {auc:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Insertion curve
        if insertion_curve:
            x, y = zip(*insertion_curve)
            ax2.plot(x, y, 'g-', linewidth=2, marker='o')
            ax2.fill_between(x, min(y), y, alpha=0.3)
            ax2.set_xlabel('Percentage of Patches Revealed')
            ax2.set_ylabel('Log Probability')
            ax2.set_title('Insertion Curve')
            ax2.grid(True, alpha=0.3)
            
            # Add AUC
            auc = self.compute_auc(insertion_curve)
            ax2.text(0.1, 0.9, f'AUC: {auc:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


class SanityChecker:
    """Sanity checks for attention visualization"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
    
    def checkerboard_test(self, bright_quadrant='upper_left'):
        """
        Create checkerboard image and verify attention concentrates in bright quadrant
        """
        # Create test image
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if bright_quadrant == 'upper_left':
            test_img[:112, :112] = 255
        elif bright_quadrant == 'upper_right':
            test_img[:112, 112:] = 255
        elif bright_quadrant == 'lower_left':
            test_img[112:, :112] = 255
        elif bright_quadrant == 'lower_right':
            test_img[112:, 112:] = 255
        
        # Convert to PIL
        from PIL import Image
        pil_image = Image.fromarray(test_img)
        
        # Get attention (you'll need to import your attention extraction function)
        # For now, returning placeholder
        print(f"Checkerboard test with bright {bright_quadrant}")
        print("TODO: Integrate with main attention extraction")
        
        # Expected: attention should concentrate in bright quadrant
        return True
    
    def model_randomization_test(self, image, prompt, num_layers=5):
        """
        Progressively randomize vision layers and check attention degradation
        """
        print(f"Model randomization test on {num_layers} layers")
        print("TODO: Implement layer randomization")
        
        # Expected: correlation should decrease as more layers randomized
        return []
    
    def label_randomization_test(self, image, num_words=5):
        """
        Use random target words and check for spurious correlations
        """
        random_words = ["pneumonia", "car", "banana", "quantum", "happiness"][:num_words]
        
        print(f"Label randomization test with words: {random_words}")
        print("TODO: Compute attention for each word and check correlations")
        
        # Expected: low correlation between different random words
        return True


def run_validation_suite(model, processor, image, prompt, target_word, attention_map):
    """Run complete validation suite"""
    print("="*60)
    print("Running Faithfulness Validation Suite")
    print("="*60)
    
    validator = FaithfulnessValidator(model, processor)
    
    # 1. Deletion curve
    print("\n1. Computing Deletion Curve...")
    deletion_curve = validator.deletion_curve(image, attention_map, prompt, target_word)
    deletion_auc = validator.compute_auc(deletion_curve)
    print(f"Deletion AUC: {deletion_auc:.3f}")
    
    # 2. Insertion curve
    print("\n2. Computing Insertion Curve...")
    insertion_curve = validator.insertion_curve(image, attention_map, prompt, target_word)
    insertion_auc = validator.compute_auc(insertion_curve)
    print(f"Insertion AUC: {insertion_auc:.3f}")
    
    # 3. Comprehensiveness & Sufficiency
    print("\n3. Computing Comprehensiveness & Sufficiency...")
    comp, suff = validator.comprehensiveness_sufficiency(image, attention_map, prompt, target_word)
    print(f"Comprehensiveness: {comp:.3f}")
    print(f"Sufficiency: {suff:.3f}")
    
    # 4. Visualize
    fig = validator.visualize_curves(deletion_curve, insertion_curve)
    
    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Deletion AUC:        {deletion_auc:.3f} {'✅' if deletion_auc > 0.5 else '⚠️'}")
    print(f"Insertion AUC:       {insertion_auc:.3f} {'✅' if insertion_auc > 0.4 else '⚠️'}")
    print(f"Comprehensiveness:   {comp:.3f} {'✅' if comp > 0.3 else '⚠️'}")
    print(f"Sufficiency:         {suff:.3f} {'✅' if suff > 0.2 else '⚠️'}")
    
    return {
        'deletion_curve': deletion_curve,
        'insertion_curve': insertion_curve,
        'deletion_auc': deletion_auc,
        'insertion_auc': insertion_auc,
        'comprehensiveness': comp,
        'sufficiency': suff,
        'figure': fig
    }


if __name__ == "__main__":
    print("Faithfulness Validation Module")
    print("This module provides quantitative validation of attention maps")
    print("\nUsage:")
    print("  from faithfulness_validation import run_validation_suite")
    print("  results = run_validation_suite(model, processor, image, prompt, target_word, attention_map)")
    print("\nMetrics computed:")
    print("  - Deletion curve & AUC")
    print("  - Insertion curve & AUC")
    print("  - Comprehensiveness")
    print("  - Sufficiency")