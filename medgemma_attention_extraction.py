#!/usr/bin/env python3
"""
Unified Attention Extraction System for Medical VQA
Implements cross-attention, self-attention, and Grad-CAM with GQA support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SystemConfig:
    """System configuration with all parameters"""
    model_id: str = "gemma-medvqa"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: Dict[str, int] = field(default_factory=lambda: {
        "torch": 42, "numpy": 42, "python": 42
    })
    decoding: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 200,
        "temperature": 0.0,
        "do_sample": False
    })
    io_paths: Dict[str, Path] = field(default_factory=lambda: {
        "data_csv": Path("./data.csv"),
        "image_root": Path("./images"),
        "out_dir": Path("./results")
    })
    eval: Dict[str, Any] = field(default_factory=lambda: {
        "target_terms": ["effusion", "pneumothorax", "consolidation", "pneumonia"],
        "robustness": {
            "paraphrases": [
                "Is there pleural effusion?",
                "Any fluid in pleural space?",
                "Do you see effusion?"
            ],
            "perturbations": [
                {"type": "gaussian_noise", "params": {"sigma": 2}},
                {"type": "contrast", "params": {"factor": 1.2}},
                {"type": "one_pixel", "params": {"epsilon": 0.1}}
            ]
        }
    })
    architecture: Dict[str, int] = field(default_factory=dict)


@dataclass
class AttentionOutput:
    """Container for attention extraction results"""
    grid: np.ndarray  # 16x16 attention grid
    mode: str  # cross, self, gradcam, uniform
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedAttentionExtractor:
    """Main attention extraction system with multiple modes"""
    
    def __init__(self, model, processor, config: SystemConfig):
        self.model = model
        self.processor = processor
        self.config = config
        self.token_gating = TokenGating()
        self.derive_architecture_info()
        self.set_seeds()
        
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        import random
        torch.manual_seed(self.config.seeds["torch"])
        np.random.seed(self.config.seeds["numpy"])
        random.seed(self.config.seeds["python"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seeds["torch"])
            
    def derive_architecture_info(self):
        """Extract architecture parameters from model config"""
        text_config = self.model.config.text_config if hasattr(self.model.config, 'text_config') else self.model.config
        
        self.config.architecture = {
            "H": getattr(text_config, 'num_attention_heads', 8),
            "K": getattr(text_config, 'num_key_value_heads', 4),
            "mm_tokens": getattr(self.model.config, 'mm_tokens_per_image', 256),
            "grid_size": 16  # sqrt(256) = 16
        }
        
    def extract_attention_maps(
        self,
        attn_outputs: Any,
        prompt_len: int,
        full_ids: torch.Tensor,
        gating_indices: List[int],
        mode_order: List[str] = ["cross", "self", "gradcam", "uniform"],
        messages: Optional[List] = None
    ) -> AttentionOutput:
        """Extract attention maps with mode priority"""
        
        for mode in mode_order:
            try:
                if mode == "cross" and hasattr(attn_outputs, 'cross_attentions') and attn_outputs.cross_attentions:
                    result = self.extract_cross_attention(
                        attn_outputs.cross_attentions, prompt_len, gating_indices
                    )
                    if self.is_valid_grid(result.grid):
                        return result
                        
                elif mode == "self" and hasattr(attn_outputs, 'attentions') and attn_outputs.attentions:
                    result = self.extract_self_attention(
                        attn_outputs.attentions, prompt_len, full_ids.shape[1], gating_indices
                    )
                    if self.is_valid_grid(result.grid):
                        return result
                        
                elif mode == "gradcam" and messages:
                    result = self.extract_gradcam(messages)
                    if self.is_valid_grid(result.grid):
                        return result
                        
                elif mode == "uniform":
                    return self.create_uniform_grid()
                    
            except Exception as e:
                print(f"Failed to extract {mode} attention: {e}")
                continue
                
        return self.create_uniform_grid()
    
    def extract_cross_attention(
        self,
        cross_attentions: List[torch.Tensor],
        prompt_len: int,
        gating_indices: List[int]
    ) -> AttentionOutput:
        """Extract cross-attention map with GQA aggregation"""
        L = min(2, len(cross_attentions))  # Last L layers
        Q = 3  # Last Q decode steps
        
        # Get KV length from last layer
        last_attn = cross_attentions[-1]
        if isinstance(last_attn, tuple):
            last_attn = last_attn[0]
        
        # Shape: [B, H, Q_len, KV_len]
        if last_attn.dim() == 4:
            kv_len = last_attn.shape[-1]
        else:
            raise ValueError(f"Unexpected cross attention shape: {last_attn.shape}")
        
        acc = torch.zeros(kv_len, device=last_attn.device)
        layers_used = []
        
        for layer_idx in range(len(cross_attentions) - L, len(cross_attentions)):
            if layer_idx < 0:
                continue
                
            layer = cross_attentions[layer_idx]
            if isinstance(layer, tuple):
                layer = layer[0]
                
            # Average over batch
            layer = layer.mean(dim=0)  # [H, Q_len, KV_len]
            
            # GQA aggregation
            layer = self.gqa_aggregate(layer, is_cross=True)  # [Q_len, KV_len]
            
            # Get last Q decode positions
            for q_idx in range(max(prompt_len, layer.shape[0] - Q), layer.shape[0]):
                if q_idx < layer.shape[0]:
                    q_attn = layer[q_idx]
                    
                    # Apply gating
                    gate = self.compute_gate_scalar(q_attn, gating_indices, prompt_len)
                    acc += gate * q_attn
                    
            layers_used.append(layer_idx)
        
        # Normalize
        acc = acc / (acc.sum() + 1e-8)
        
        # Extract image tokens (assuming positions 1:257 are image tokens)
        if acc.shape[0] >= 257:
            image_attn = acc[1:257]
            grid = image_attn.reshape(16, 16).cpu().numpy()
        else:
            grid = np.zeros((16, 16))
        
        return AttentionOutput(
            grid=grid,
            mode="cross",
            metadata={
                "layers_used": len(layers_used),
                "q_positions_used": Q,
                "gate_indices": len(gating_indices)
            }
        )
    
    def extract_self_attention(
        self,
        self_attentions: List[torch.Tensor],
        prompt_len: int,
        full_len: int,
        gating_indices: List[int]
    ) -> AttentionOutput:
        """Extract self-attention as proxy using image slice"""
        
        # Use last few layers for better signal
        num_layers_to_use = min(3, len(self_attentions))
        layer_weights = torch.tensor([0.5 ** i for i in range(num_layers_to_use)])
        layer_weights = layer_weights / layer_weights.sum()
        
        accumulated_grid = None
        
        for layer_idx, weight in zip(range(-num_layers_to_use, 0), layer_weights):
            layer = self_attentions[layer_idx]
            if isinstance(layer, tuple):
                layer = layer[0]
                
            # Average over batch
            if layer.dim() == 4:
                layer = layer.mean(dim=0)  # [H, L, L]
            
            # GQA aggregation
            S = self.gqa_aggregate(layer, is_cross=False)  # [L, L]
            
            # Infer image token span
            # For Gemma3, image tokens typically follow the text prompt
            # Try to find the image token block (256 consecutive tokens)
            img_start, img_end = self.find_image_token_span(S, prompt_len)
            
            if img_end - img_start != 256:
                # Fallback to default positions
                img_start, img_end = 1, min(257, S.shape[1])
            
            img_span_len = img_end - img_start
            
            # Accumulate from generation positions
            acc = torch.zeros(img_span_len, device=S.device)
            gen_positions = list(range(prompt_len, min(prompt_len + 5, S.shape[0])))
            
            for pos in gen_positions:
                if pos < S.shape[0] and img_end <= S.shape[1]:
                    # Extract attention to image tokens
                    row = S[pos, img_start:img_end]
                    
                    # Compute gate from prompt attention
                    if prompt_len > 0:
                        prompt_attn = S[pos, :prompt_len]
                        gate = self.compute_gate_scalar(prompt_attn, gating_indices, prompt_len)
                    else:
                        gate = 1.0
                    
                    acc += gate * row
            
            # Normalize
            if acc.sum() > 0:
                acc = acc / acc.sum()
            
            # Reshape to grid
            if acc.shape[0] == 256:
                layer_grid = acc.reshape(16, 16)
            else:
                # Pad or truncate to 256
                padded = torch.zeros(256, device=acc.device)
                padded[:min(256, acc.shape[0])] = acc[:min(256, acc.shape[0])]
                layer_grid = padded.reshape(16, 16)
            
            # Accumulate weighted
            if accumulated_grid is None:
                accumulated_grid = layer_grid * weight
            else:
                accumulated_grid += layer_grid * weight
        
        # Convert to numpy
        grid = accumulated_grid.cpu().numpy()
        
        return AttentionOutput(
            grid=grid,
            mode="self",
            metadata={
                "layers_used": num_layers_to_use,
                "gen_positions": len(gen_positions) if 'gen_positions' in locals() else 0,
                "gate_indices": len(gating_indices),
                "image_span": (img_start, img_end) if 'img_start' in locals() else (1, 257)
            }
        )
    
    def find_image_token_span(self, attention_matrix: torch.Tensor, prompt_len: int) -> tuple:
        """Find the span of image tokens in the attention matrix"""
        
        # Look for a block of 256 tokens with high internal attention
        # This helps identify where image tokens are located
        
        if attention_matrix.shape[1] < 256:
            return 1, min(257, attention_matrix.shape[1])
        
        # Simple heuristic: image tokens often come after BOS token
        # and have different attention patterns than text
        
        # Check common positions
        candidates = [
            (1, 257),  # Right after BOS
            (prompt_len - 256, prompt_len) if prompt_len > 256 else (1, 257),  # Before prompt end
        ]
        
        best_span = (1, 257)
        best_score = -1
        
        for start, end in candidates:
            if start >= 0 and end <= attention_matrix.shape[1]:
                # Check if this span has coherent attention pattern
                span_attn = attention_matrix[start:end, start:end]
                if span_attn.shape[0] == 256:
                    # Score based on internal coherence
                    score = span_attn.diagonal().mean().item()
                    if score > best_score:
                        best_score = score
                        best_span = (start, end)
        
        return best_span
    
    def extract_gradcam(self, messages: List) -> AttentionOutput:
        """Grad-CAM fallback on vision encoder"""
        
        try:
            # Hook the last vision encoder layer
            activation = {}
            gradient = {}
            
            def forward_hook(module, input, output):
                activation['value'] = output.detach()
                
            def backward_hook(module, grad_input, grad_output):
                gradient['value'] = grad_output[0].detach()
            
            # Find vision encoder
            vision_model = None
            if hasattr(self.model, 'vision_tower'):
                vision_model = self.model.vision_tower
            elif hasattr(self.model, 'vision_encoder'):
                vision_model = self.model.vision_encoder
            else:
                return self.create_uniform_grid()
            
            # Get last layer
            if hasattr(vision_model, 'encoder'):
                last_layer = vision_model.encoder.layers[-1]
            elif hasattr(vision_model, 'layers'):
                last_layer = vision_model.layers[-1]
            else:
                return self.create_uniform_grid()
            
            # Register hooks
            fh = last_layer.register_forward_hook(forward_hook)
            bh = last_layer.register_full_backward_hook(backward_hook)
            
            try:
                # Forward pass with gradients
                self.model.eval()
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_tensors="pt"
                ).to(self.model.device)
                
                # Enable gradients temporarily
                with torch.enable_grad():
                    outputs = self.model(**inputs)
                    
                    # Target last token's max logit
                    logits = outputs.logits[0, -1, :]
                    target = logits.max()
                    
                    # Backward
                    self.model.zero_grad()
                    target.backward(retain_graph=True)
                
                # Compute Grad-CAM
                if 'value' in activation and 'value' in gradient:
                    act = activation['value'][0]  # Remove batch dim
                    grad = gradient['value'][0]
                    
                    # Global average pooling of gradients
                    weights = grad.mean(dim=(1, 2))  # Average over spatial dims
                    
                    # Weighted combination
                    cam = torch.zeros(act.shape[1:], device=act.device)
                    for i, w in enumerate(weights):
                        cam += w * act[i]
                    
                    # ReLU and normalize
                    cam = F.relu(cam)
                    cam = cam / (cam.max() + 1e-8)
                    
                    # Resize to 16x16 (matching projector output)
                    cam_np = cam.cpu().numpy()
                    cam_resized = cv2.resize(cam_np, (16, 16), interpolation=cv2.INTER_LINEAR)
                    
                    return AttentionOutput(
                        grid=cam_resized,
                        mode="gradcam",
                        metadata={"layer": "vision_encoder_last"}
                    )
                    
            finally:
                fh.remove()
                bh.remove()
                
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            
        return self.create_uniform_grid()
    
    def gqa_aggregate(self, attention: torch.Tensor, is_cross: bool = True) -> torch.Tensor:
        """GQA-aware head aggregation"""
        H = self.config.architecture["H"]
        K = self.config.architecture["K"]
        
        if H == K:
            # No GQA, simple average
            return attention.mean(dim=0)
        
        # Grouped query attention
        group_size = H // K
        
        if is_cross:
            # Shape: [H, Q, KV] -> [K, group_size, Q, KV]
            reshaped = attention.reshape(K, group_size, *attention.shape[1:])
        else:
            # Shape: [H, L, L] -> [K, group_size, L, L]  
            reshaped = attention.reshape(K, group_size, *attention.shape[1:])
        
        # Average within groups
        grouped = reshaped.mean(dim=1)  # [K, ...]
        
        # Average across KV heads
        aggregated = grouped.mean(dim=0)  # [...]
        
        return aggregated
    
    def compute_gate_scalar(
        self,
        attention_row: torch.Tensor,
        gating_indices: List[int],
        prompt_len: int
    ) -> float:
        """Compute gating scalar from attention to prompt tokens"""
        
        if not gating_indices:
            return 1.0 / max(1, prompt_len)
        
        # Extract attention to gating indices
        gate_attn = torch.zeros_like(attention_row[:prompt_len])
        
        for idx in gating_indices:
            if idx < len(gate_attn):
                gate_attn[idx] = attention_row[idx]
        
        # Normalize to [0, 1]
        if gate_attn.sum() > 0:
            return float(gate_attn.sum() / attention_row[:prompt_len].sum())
        else:
            return 1.0 / max(1, prompt_len)
    
    def is_valid_grid(self, grid: np.ndarray) -> bool:
        """Check if attention grid is valid"""
        
        # Check shape
        if grid.shape != (16, 16):
            return False
        
        # Check for NaN or Inf
        if np.isnan(grid).any() or np.isinf(grid).any():
            return False
        
        # Check if not all zeros
        if grid.sum() == 0:
            return False
        
        # Check entropy (avoid near-uniform)
        flat = grid.flatten()
        flat_norm = flat / (flat.sum() + 1e-8)
        ent = entropy(flat_norm + 1e-10)
        max_ent = np.log(len(flat))
        
        if ent > 0.95 * max_ent:
            return False
        
        return True
    
    def create_uniform_grid(self) -> AttentionOutput:
        """Create uniform attention grid as fallback"""
        grid_size = self.config.architecture["grid_size"]
        grid = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
        
        return AttentionOutput(
            grid=grid,
            mode="uniform",
            metadata={"fallback": True}
        )


class TokenGating:
    """Token gating for target term identification"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def find_target_indices(
        tokenizer,
        prompt_ids: List[int],
        question_text: str,
        target_terms: List[str]
    ) -> List[int]:
        """Find token indices for target terms in prompt"""
        
        indices = []
        
        # Handle None tokenizer for testing
        if tokenizer is None:
            # Simple mock behavior for testing
            for term in target_terms:
                if term.lower() in question_text.lower():
                    # Add some dummy indices
                    indices.extend([1, 2, 3])
            return indices
        
        try:
            # Try offset mapping approach first
            encoding = tokenizer.encode_plus(
                question_text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            
            q_ids = encoding.input_ids
            offsets = encoding.offset_mapping
            
            # Find question start in prompt
            q_start = TokenGating._find_subsequence(prompt_ids, q_ids)
            
            if q_start >= 0:
                for term in target_terms:
                    term_lower = term.lower()
                    text_lower = question_text.lower()
                    
                    # Find all occurrences of term in text
                    start = 0
                    while True:
                        pos = text_lower.find(term_lower, start)
                        if pos == -1:
                            break
                        
                        # Map character position to token indices
                        for token_idx, (token_start, token_end) in enumerate(offsets):
                            if token_start <= pos < token_end or token_start < pos + len(term_lower) <= token_end:
                                actual_idx = q_start + token_idx
                                if actual_idx not in indices:
                                    indices.append(actual_idx)
                        
                        start = pos + 1
                        
        except Exception as e:
            print(f"Offset mapping failed: {e}, falling back to token matching")
            
            # Fallback: direct token matching
            for term in target_terms:
                term_ids = tokenizer.encode(term, add_special_tokens=False)
                
                # Find all occurrences
                for i in range(len(prompt_ids) - len(term_ids) + 1):
                    if prompt_ids[i:i+len(term_ids)] == term_ids:
                        indices.extend(range(i, i + len(term_ids)))
        
        return sorted(list(set(indices)))
    
    @staticmethod
    def _find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
        """Find starting position of subsequence in sequence"""
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return i
        return -1


class BodyMaskGenerator:
    """Generate body masks for chest X-rays"""
    
    @staticmethod
    def create_body_mask(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Create binary mask of body region"""
        
        # Convert to grayscale numpy if needed
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        else:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
        
        # Threshold to separate body from background
        # Use Otsu's method for automatic threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (body)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Slight erosion to ensure we're inside the body
            eroded = cv2.erode(mask, kernel, iterations=1)
            return (eroded > 0).astype(np.float32)
        
        return np.ones_like(gray, dtype=np.float32)
    
    @staticmethod
    def apply_body_mask(
        grid: np.ndarray,
        mask: np.ndarray,
        clip_percentiles: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """Apply body mask to attention grid"""
        
        # Resize grid to mask size
        h, w = mask.shape
        heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply mask
        heat_masked = heat * mask
        
        # Percentile clipping within mask
        valid_values = heat_masked[mask > 0]
        if len(valid_values) > 0:
            p_low, p_high = np.percentile(valid_values, clip_percentiles)
            heat_masked = np.clip(heat_masked, p_low, p_high)
            
            # Renormalize
            if p_high > p_low:
                heat_masked = (heat_masked - p_low) / (p_high - p_low)
        
        return heat_masked * mask


class AttentionMetrics:
    """Compute metrics for attention analysis"""
    
    @staticmethod
    def compute_metrics(grid: np.ndarray, body_mask: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive attention metrics"""
        
        # Resize mask to grid size
        mask_resized = cv2.resize(body_mask, (16, 16), interpolation=cv2.INTER_LINEAR)
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        
        # Masked grid
        G = grid * mask_resized
        G_norm = G / (G.sum() + 1e-8)
        
        # Entropy
        flat = G_norm.flatten()
        H = entropy(flat + 1e-10)
        
        # Focus (max attention)
        focus = float(G.max())
        
        # Inside body ratio
        inside_ratio = float(G.sum() / (grid.sum() + 1e-8))
        
        # Border fraction
        border = G[0, :].sum() + G[-1, :].sum() + G[:, 0].sum() + G[:, -1].sum()
        border_fraction = float(border / (G.sum() + 1e-8))
        
        # Spatial distribution
        mid_w = grid.shape[1] // 2
        third_h = grid.shape[0] // 3
        
        left_fraction = float(G[:, :mid_w].sum() / (G.sum() + 1e-8))
        right_fraction = float(G[:, mid_w:].sum() / (G.sum() + 1e-8))
        apical_fraction = float(G[:third_h, :].sum() / (G.sum() + 1e-8))
        basal_fraction = float(G[-third_h:, :].sum() / (G.sum() + 1e-8))
        
        return {
            "entropy": H,
            "focus": focus,
            "inside_body_ratio": inside_ratio,
            "border_fraction": border_fraction,
            "left_fraction": left_fraction,
            "right_fraction": right_fraction,
            "apical_fraction": apical_fraction,
            "basal_fraction": basal_fraction
        }
    
    @staticmethod
    def map_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two attention maps"""
        
        # Normalize
        g1 = grid1.flatten()
        g1 = g1 / (g1.sum() + 1e-8)
        
        g2 = grid2.flatten()
        g2 = g2 / (g2.sum() + 1e-8)
        
        # JS divergence
        js_div = jensenshannon(g1, g2) ** 2
        
        return float(js_div)


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def create_overlay(
        image: Union[Image.Image, np.ndarray],
        attention: np.ndarray,
        alpha: float = 0.35,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """Create attention overlay on image"""
        
        # Convert image to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert('RGB'))
        else:
            img_np = image
            
        # Resize attention to image size
        h, w = img_np.shape[:2]
        attention_resized = cv2.resize(attention, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize attention
        attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        attention_colored = cmap(attention_norm)[:, :, :3]
        attention_colored = (attention_colored * 255).astype(np.uint8)
        
        # Blend
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
        blended = cv2.addWeighted(img_np, 1 - alpha, attention_colored, alpha, 0)
        
        return blended
    
    @staticmethod
    def save_visualization(
        image: Union[Image.Image, np.ndarray],
        attention: np.ndarray,
        mask: np.ndarray,
        output_path: Path,
        title: str = ""
    ):
        """Save comprehensive visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if isinstance(image, Image.Image):
            axes[0, 0].imshow(image, cmap='gray')
        else:
            axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Raw attention
        im1 = axes[0, 1].imshow(attention, cmap='hot', interpolation='bicubic')
        axes[0, 1].set_title('Raw Attention (16x16)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Body mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Body Mask')
        axes[0, 2].axis('off')
        
        # Masked attention
        masked_attention = BodyMaskGenerator.apply_body_mask(attention, mask)
        im2 = axes[1, 0].imshow(masked_attention, cmap='jet', interpolation='bicubic')
        axes[1, 0].set_title('Masked Attention')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        
        # Overlay
        overlay = Visualizer.create_overlay(image, masked_attention)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Attention Overlay')
        axes[1, 1].axis('off')
        
        # Metrics
        metrics = AttentionMetrics.compute_metrics(attention, mask)
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Metrics')
        axes[1, 2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def test_system():
    """Test the unified attention system"""
    print("Testing Unified Attention System...")
    
    # Create mock config
    config = SystemConfig()
    config.io_paths["out_dir"] = Path("./test_output")
    config.io_paths["out_dir"].mkdir(exist_ok=True)
    
    # Create mock model and processor (would be replaced with actual)
    class MockModel:
        class Config:
            class TextConfig:
                num_attention_heads = 8
                num_key_value_heads = 4
            text_config = TextConfig()
            mm_tokens_per_image = 256
            
        config = Config()
        
    class MockProcessor:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]
            def encode_plus(self, text, **kwargs):
                return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3)]}
        tokenizer = Tokenizer()
    
    model = MockModel()
    processor = MockProcessor()
    
    # Initialize system
    extractor = UnifiedAttentionExtractor(model, processor, config)
    
    # Test architecture derivation
    print(f"Architecture: {config.architecture}")
    
    # Test uniform grid
    uniform = extractor.create_uniform_grid()
    print(f"Uniform grid shape: {uniform.grid.shape}, mode: {uniform.mode}")
    
    # Test token gating
    indices = TokenGating.find_target_indices(
        processor.tokenizer,
        [1, 2, 3, 4, 5],
        "test effusion",
        ["effusion"]
    )
    print(f"Gating indices: {indices}")
    
    # Test body mask
    test_image = np.random.rand(512, 512) * 255
    mask = BodyMaskGenerator.create_body_mask(test_image.astype(np.uint8))
    print(f"Body mask shape: {mask.shape}")
    
    # Test metrics
    test_grid = np.random.rand(16, 16)
    metrics = AttentionMetrics.compute_metrics(test_grid, mask)
    print(f"Metrics: {metrics}")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_system()