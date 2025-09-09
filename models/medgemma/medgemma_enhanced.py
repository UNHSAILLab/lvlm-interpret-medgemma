#!/usr/bin/env python3
"""
Enhanced MedGemma 4B Multimodal VLM with Robust Attention Extraction
=====================================================================
Comprehensive improvements for attention visualization with multiple fallback modes.
"""

import gc
import logging
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths configuration
MIMIC_CSV_PATH = Path('/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv')
MIMIC_IMAGE_BASE_PATH = Path('/home/bsada1/mimic_cxr_hundred_vqa')

# ============================================================================
# ENHANCED ATTENTION EXTRACTION WITH MODE SELECTOR
# ============================================================================

def extract_token_conditioned_attention_enhanced(
    outputs,
    inputs,
    processor,
    target_prompt_indices: Optional[List[int]] = None,
    mode: str = 'auto',
    model=None,
    question_text: Optional[str] = None,
    target_words: Optional[List[str]] = None,
    use_group_aware: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhanced attention extraction with multiple modes and fallbacks.
    
    Args:
        outputs: Model outputs with attention weights
        inputs: Model inputs including pixel_values
        processor: Tokenizer/processor for the model
        target_prompt_indices: Indices of target tokens in prompt
        mode: 'auto', 'cross', 'self', 'gradcam', or 'uniform'
        model: Model instance (required for gradcam)
        question_text: Original question text for better token matching
        target_words: Target words to focus on
        use_group_aware: Use group-aware head aggregation for GQA
        
    Returns:
        attention_grid: 2D attention map
        metadata: Dictionary with extraction details
    """
    
    metadata = {
        'mode_requested': mode,
        'mode_used': None,
        'fallback_chain': [],
        'grid_shape': None,
        'target_indices': [],
        'diagnostics': {}
    }
    
    # Improve target gating if needed
    if target_prompt_indices is None and question_text and target_words:
        target_prompt_indices = find_prompt_span_and_target_indices(
            processor.tokenizer, 
            inputs['input_ids'][0].cpu().tolist(),
            question_text,
            target_words
        )
        metadata['target_indices'] = target_prompt_indices
        logger.info(f"Found {len(target_prompt_indices) if target_prompt_indices else 0} target indices")
    
    # Define extraction methods
    methods = {
        'cross': lambda: extract_cross_attention(
            outputs, inputs, target_prompt_indices, use_group_aware
        ),
        'self': lambda: extract_self_attention(
            outputs, inputs, target_prompt_indices, use_group_aware
        ),
        'gradcam': lambda: extract_gradcam_attention(
            outputs, inputs, model, target_prompt_indices
        ),
        'uniform': lambda: extract_uniform_attention(inputs)
    }
    
    # Determine method order based on mode
    if mode == 'auto':
        method_order = ['cross', 'self', 'gradcam', 'uniform']
    elif mode in methods:
        method_order = [mode] + [m for m in ['cross', 'self', 'gradcam', 'uniform'] if m != mode]
    else:
        logger.warning(f"Unknown mode {mode}, using auto")
        method_order = ['cross', 'self', 'gradcam', 'uniform']
    
    # Try methods in order
    for method_name in method_order:
        try:
            logger.info(f"Attempting {method_name} attention extraction...")
            metadata['fallback_chain'].append(method_name)
            
            attention_grid, method_meta = methods[method_name]()
            
            # Validate result
            if attention_grid is not None and validate_attention_grid(attention_grid):
                metadata['mode_used'] = method_name
                metadata['grid_shape'] = attention_grid.shape
                metadata['diagnostics'].update(method_meta)
                logger.info(f"Successfully extracted attention using {method_name} mode")
                return attention_grid, metadata
                
        except Exception as e:
            logger.warning(f"{method_name} extraction failed: {e}")
            continue
    
    # Final fallback - should never reach here
    logger.error("All attention extraction methods failed!")
    return np.ones((16, 16)) / 256, metadata


def extract_cross_attention(outputs, inputs, target_indices, use_group_aware=True):
    """Extract cross-attention with group-aware aggregation."""
    
    if not hasattr(outputs, 'cross_attentions') or outputs.cross_attentions is None:
        raise ValueError("No cross-attention weights available")
    
    # Get cross attention from last layer
    cross_attn = outputs.cross_attentions[-1]  # Shape: [batch, heads, seq_len, kv_len]
    
    if cross_attn is None or len(cross_attn.shape) != 4:
        raise ValueError("Invalid cross-attention shape")
    
    # Aggregate across heads (group-aware if needed)
    if use_group_aware:
        attn_map = aggregate_heads_group_aware(cross_attn)
    else:
        attn_map = cross_attn.mean(dim=1)  # Average over heads
    
    # Apply target gating if available
    if target_indices:
        gating = torch.zeros(attn_map.shape[1])
        gating[target_indices] = 1.0
        attn_map = attn_map * gating.unsqueeze(0).unsqueeze(-1)
    
    # Extract image region attention
    image_attn = extract_image_attention(attn_map, inputs)
    
    # Reshape to grid
    grid_size = int(np.sqrt(image_attn.shape[-1]))
    attention_grid = image_attn.reshape(grid_size, grid_size)
    
    metadata = {
        'cross_attn_shape': list(cross_attn.shape),
        'num_heads': cross_attn.shape[1],
        'used_group_aware': use_group_aware
    }
    
    return attention_grid.cpu().numpy(), metadata


def extract_self_attention(outputs, inputs, target_indices, use_group_aware=True):
    """Extract self-attention focusing on image tokens."""
    
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        raise ValueError("No self-attention weights available")
    
    # Get self attention from last layer
    self_attn = outputs.attentions[-1]  # Shape: [batch, heads, seq_len, seq_len]
    
    if self_attn is None or len(self_attn.shape) != 4:
        raise ValueError("Invalid self-attention shape")
    
    # Build self-attention grid for image tokens
    attention_grid, metadata = build_self_attention_grid(
        outputs, inputs, target_indices, use_group_aware
    )
    
    return attention_grid, metadata


def build_self_attention_grid(outputs, inputs, token_indices, use_group_aware=True):
    """
    Build attention grid from self-attention weights.
    
    Focuses on how generated tokens attend to image patches.
    """
    
    # Get last layer self-attention
    self_attn = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = self_attn.shape
    
    # Infer image token positions
    # Typically: [BOS, image_tokens..., text_tokens...]
    # Need to identify the image token span
    
    pixel_values = inputs.get('pixel_values')
    if pixel_values is not None:
        H, W = pixel_values.shape[-2:]
        patch_size = 16  # Common for vision transformers
        grid_h, grid_w = H // patch_size, W // patch_size
        num_image_tokens = grid_h * grid_w
    else:
        # Fallback: assume 256 image tokens (16x16 grid)
        num_image_tokens = 256
        grid_h = grid_w = 16
    
    # Find image token span (usually after BOS token)
    # This is model-specific; for MedGemma, image tokens typically start at position 1
    image_start = 1
    image_end = image_start + num_image_tokens
    
    # Extract attention from generated tokens to image tokens
    # Use the last few tokens (the generated answer)
    num_generated = min(20, seq_len - image_end)  # Last N tokens
    if num_generated <= 0:
        raise ValueError("No generated tokens found")
    
    generated_start = seq_len - num_generated
    
    # Get attention weights: [batch, heads, generated_tokens, image_tokens]
    attn_to_image = self_attn[:, :, generated_start:, image_start:image_end]
    
    # Aggregate across generated tokens (average)
    attn_to_image = attn_to_image.mean(dim=2)  # [batch, heads, image_tokens]
    
    # Aggregate across heads (group-aware if needed)
    if use_group_aware:
        attn_map = aggregate_heads_group_aware(attn_to_image.unsqueeze(2)).squeeze(2)
    else:
        attn_map = attn_to_image.mean(dim=1)  # [batch, image_tokens]
    
    # Reshape to grid
    attention_grid = attn_map[0].reshape(grid_h, grid_w)
    
    metadata = {
        'self_attn_shape': list(self_attn.shape),
        'num_heads': num_heads,
        'num_image_tokens': num_image_tokens,
        'grid_shape': (grid_h, grid_w),
        'generated_tokens_used': num_generated,
        'used_group_aware': use_group_aware
    }
    
    return attention_grid.cpu().numpy(), metadata


def extract_gradcam_attention(outputs, inputs, model, target_indices):
    """Extract attention using Grad-CAM with proper gradient enabling."""
    
    if model is None:
        raise ValueError("Model required for Grad-CAM")
    
    # Enable gradients temporarily
    original_grad_state = {}
    for name, param in model.named_parameters():
        original_grad_state[name] = param.requires_grad
        param.requires_grad = True
    
    try:
        with torch.set_grad_enabled(True):
            with torch.autocast('cuda', enabled=False):
                attention_grid, metadata = gradcam_on_vision_enhanced(
                    model, inputs, outputs
                )
    finally:
        # Restore original gradient state
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_state[name]
    
    return attention_grid, metadata


def gradcam_on_vision_enhanced(model, inputs, outputs):
    """
    Enhanced Grad-CAM implementation with robust layer selection.
    """
    
    # Find vision encoder layers
    vision_tower = None
    for name, module in model.named_modules():
        if 'vision_tower' in name and hasattr(module, 'vision_model'):
            vision_tower = module.vision_model
            break
    
    if vision_tower is None:
        raise ValueError("Could not find vision tower")
    
    # Get encoder layers
    encoder_layers = vision_tower.encoder.layers
    
    # Use last layer or second-to-last as fallback
    target_layer_idx = -1
    target_layer = encoder_layers[target_layer_idx]
    logger.info(f"Using vision encoder layer {target_layer_idx}")
    
    # Hook for capturing activations
    activations = {}
    def hook_fn(module, input, output):
        activations['output'] = output
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Forward pass with gradient computation
        outputs_with_grad = model(**inputs)
        
        # Get activations
        if 'output' not in activations:
            raise ValueError("Failed to capture activations")
        
        layer_output = activations['output']
        
        # Compute gradients with respect to output logits
        logits = outputs_with_grad.logits
        target_logit = logits[0, -1, :].max()  # Max logit of last generated token
        
        # Backward pass
        model.zero_grad()
        target_logit.backward(retain_graph=True)
        
        # Get gradients
        gradients = layer_output.grad
        
        if gradients is None:
            raise ValueError("No gradients computed")
        
        # Compute Grad-CAM
        weights = gradients.mean(dim=(0, 2))  # Global average pooling
        cam = torch.zeros(layer_output.shape[2], device=layer_output.device)
        
        for i, w in enumerate(weights[0]):
            cam += w * layer_output[0, i, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Reshape if needed
        if len(cam.shape) == 1:
            # Assume square grid
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(grid_size, grid_size)
        
        # Normalize
        cam = cam / (cam.max() + 1e-8)
        
    finally:
        hook.remove()
    
    metadata = {
        'layer_used': f"encoder_layer_{target_layer_idx}",
        'activation_shape': list(layer_output.shape),
        'gradient_computed': True
    }
    
    return cam.cpu().numpy(), metadata


def extract_uniform_attention(inputs):
    """Fallback: uniform attention across image."""
    
    # Determine grid size from inputs
    pixel_values = inputs.get('pixel_values')
    if pixel_values is not None:
        H, W = pixel_values.shape[-2:]
        grid_size = min(H, W) // 16  # Assuming patch size of 16
    else:
        grid_size = 16  # Default
    
    attention_grid = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
    
    metadata = {
        'fallback_reason': 'uniform distribution',
        'grid_size': grid_size
    }
    
    return attention_grid, metadata


def extract_image_attention(attn_map, inputs):
    """Extract attention weights corresponding to image tokens."""
    
    # Typically image tokens are in a specific range
    # For MedGemma, usually positions 1 to 257 (256 image tokens)
    
    # Get sequence length
    seq_len = attn_map.shape[-1]
    
    # Estimate image token range
    # This is model-specific; adjust as needed
    image_start = 1
    image_end = min(257, seq_len)
    
    # Extract image attention
    image_attn = attn_map[:, :, image_start:image_end]
    
    # Average over batch and sequence dimensions
    image_attn = image_attn.mean(dim=0).mean(dim=0)
    
    return image_attn


def aggregate_heads_group_aware(attention_weights):
    """
    Group-aware head aggregation for models using GQA (Grouped Query Attention).
    
    Args:
        attention_weights: Tensor of shape [batch, heads, seq_len, kv_len]
        
    Returns:
        Aggregated attention: [batch, seq_len, kv_len]
    """
    
    batch_size, num_heads, seq_len, kv_len = attention_weights.shape
    
    # Detect if using GQA based on head count
    # MedGemma uses 8 KV heads and 16 Q heads (2 groups)
    if num_heads == 16:
        num_kv_heads = 8
        group_size = num_heads // num_kv_heads
        
        # Reshape to group structure
        attention_grouped = attention_weights.reshape(
            batch_size, num_kv_heads, group_size, seq_len, kv_len
        )
        
        # Average within groups first
        attention_grouped = attention_grouped.mean(dim=2)
        
        # Then average across KV heads
        attention_aggregated = attention_grouped.mean(dim=1)
    else:
        # Standard multi-head attention
        attention_aggregated = attention_weights.mean(dim=1)
    
    return attention_aggregated


def validate_attention_grid(grid):
    """Validate that attention grid is reasonable."""
    
    if grid is None:
        return False
    
    # Check shape
    if len(grid.shape) != 2:
        return False
    
    # Check values
    if np.isnan(grid).any() or np.isinf(grid).any():
        return False
    
    # Check if not all zeros
    if grid.sum() == 0:
        return False
    
    # Check entropy (should be lower than uniform)
    grid_flat = grid.flatten()
    grid_norm = grid_flat / (grid_flat.sum() + 1e-8)
    entropy = -np.sum(grid_norm * np.log(grid_norm + 1e-8))
    
    uniform_entropy = np.log(len(grid_flat))
    if entropy >= uniform_entropy * 0.95:  # Too close to uniform
        logger.warning(f"Attention entropy too high: {entropy:.2f} vs uniform {uniform_entropy:.2f}")
        return False
    
    return True


def find_prompt_span_and_target_indices(
    tokenizer, 
    prompt_ids: List[int],
    question_text: str,
    target_words: List[str]
) -> Optional[List[int]]:
    """
    Find target token indices in the prompt using improved matching.
    
    Args:
        tokenizer: The tokenizer instance
        prompt_ids: Token IDs of the full prompt
        question_text: Original question text
        target_words: Words to find in the question
        
    Returns:
        List of token indices corresponding to target words
    """
    
    try:
        # Try fast tokenizer with offsets
        if hasattr(tokenizer, 'encode_plus'):
            encoding = tokenizer.encode_plus(
                question_text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            
            question_ids = encoding['input_ids']
            offsets = encoding.get('offset_mapping', [])
            
            # Find question span in prompt
            question_start = find_subsequence(prompt_ids, question_ids)
            
            if question_start >= 0:
                # Map target words to token indices
                target_indices = []
                
                for target_word in target_words:
                    target_lower = target_word.lower()
                    
                    # Find word position in question text
                    word_start = question_text.lower().find(target_lower)
                    if word_start >= 0:
                        word_end = word_start + len(target_word)
                        
                        # Find corresponding tokens
                        for i, (start, end) in enumerate(offsets):
                            if start >= word_start and end <= word_end:
                                global_idx = question_start + i
                                if global_idx not in target_indices:
                                    target_indices.append(global_idx)
                
                if target_indices:
                    logger.info(f"Found target indices via offset mapping: {target_indices}")
                    return target_indices
    
    except Exception as e:
        logger.warning(f"Fast tokenizer matching failed: {e}")
    
    # Fallback to robust token matching
    target_indices = []
    
    for target_word in target_words:
        # Tokenize target word
        target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
        
        # Find in prompt
        for i in range(len(prompt_ids) - len(target_tokens) + 1):
            if prompt_ids[i:i+len(target_tokens)] == target_tokens:
                target_indices.extend(range(i, i+len(target_tokens)))
    
    # Remove duplicates while preserving order
    target_indices = list(dict.fromkeys(target_indices))
    
    if target_indices:
        logger.info(f"Found target indices via token matching: {target_indices}")
        return target_indices
    
    logger.warning("No target indices found, using uniform gating")
    return None


def find_subsequence(sequence: List, subsequence: List) -> int:
    """Find starting index of subsequence in sequence, or -1 if not found."""
    
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i+len(subsequence)] == subsequence:
            return i
    return -1


# ============================================================================
# ENHANCED MODEL LOADING WITH FAST TOKENIZER
# ============================================================================

def load_model_enhanced(
    model_id='google/medgemma-4b-it', 
    device=None,
    use_fast_tokenizer=True
):
    """Load MedGemma model with enhanced configuration."""
    
    if device is None:
        device = setup_gpu()
    
    print(f"\n=== Loading MedGemma 4B Multimodal VLM ===")
    print(f"Model ID: {model_id}")
    print(f"Fast tokenizer: {use_fast_tokenizer}")
    
    # Load processor with fast tokenizer option
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=use_fast_tokenizer
    )
    print("✓ Processor loaded")
    
    try:
        # When CUDA_VISIBLE_DEVICES is set, use device 0 (which maps to the visible GPU)
        # Otherwise use the actual device index
        if device.type == 'cuda':
            # Check if CUDA_VISIBLE_DEVICES is set
            import os
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # Use device 0 since it's the only visible device after setting CUDA_VISIBLE_DEVICES
                device_idx = 0
            else:
                device_idx = device.index
        else:
            device_idx = 'cpu'
            
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': device_idx},
            attn_implementation="eager",
            tie_word_embeddings=False,
            low_cpu_mem_usage=True,
            output_attentions=True,  # Ensure attention output
            output_hidden_states=True  # For additional analysis
        )
        
        model.eval()
        
        # Configure for attention extraction
        if hasattr(model.config, 'output_attentions'):
            model.config.output_attentions = True
        if hasattr(model.config, 'output_hidden_states'):
            model.config.output_hidden_states = True
        
        print("✓ MedGemma model loaded with enhanced attention capabilities")
        
        # Log memory usage
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    return model, processor


# ============================================================================
# ENHANCED GENERATION WITH ATTENTION
# ============================================================================

def generate_with_attention_enhanced(
    model,
    processor,
    image,
    question,
    max_new_tokens=16,
    temperature=0.7,
    return_dict_in_generate=True
):
    """Generate answer with full attention outputs."""
    
    # Format question with proper image sequence
    if hasattr(processor, 'full_image_sequence'):
        # Use the full image sequence (256 image tokens + start/end)
        formatted_question = f"{processor.full_image_sequence}\n{question}"
    elif hasattr(processor, 'image_token'):
        # Fallback to single image token
        formatted_question = f"{processor.image_token}{question}"
    else:
        formatted_question = question
    
    # Prepare inputs
    inputs = processor(
        text=formatted_question,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate with attention
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=return_dict_in_generate
        )
    
    # Decode answer
    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    answer = processor.decode(generated_ids, skip_special_tokens=True)
    
    return outputs, inputs, answer


# ============================================================================
# GPU SETUP AND UTILITIES
# ============================================================================

def get_gpu_memory_from_nvidia_smi():
    """Get GPU memory info from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            gpu_info.append({
                'id': int(parts[0]),
                'total_mb': float(parts[1]),
                'used_mb': float(parts[2]),
                'free_mb': float(parts[3]),
                'free_gb': float(parts[3]) / 1024,
                'total_gb': float(parts[1]) / 1024,
                'usage_percent': (float(parts[2]) / float(parts[1])) * 100
            })
        
        return gpu_info
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def select_best_gpu(min_free_gb: float = 15.0) -> int:
    """Select GPU with most free memory."""
    gpu_info = get_gpu_memory_from_nvidia_smi()
    
    if gpu_info is None:
        print("nvidia-smi not available, using PyTorch memory info")
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = props.total_memory - torch.cuda.memory_allocated(i)
            total = props.total_memory
            
            gpu_info.append({
                'id': i,
                'free_gb': free / (1024 ** 3),
                'total_gb': total / (1024 ** 3),
                'usage_percent': (1 - free / total) * 100
            })
    
    if not gpu_info:
        raise RuntimeError("No CUDA GPUs available")
    
    print("\n=== GPU Status ===")
    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: "
              f"{gpu['free_gb']:.1f}GB free / {gpu['total_gb']:.1f}GB total "
              f"({gpu['usage_percent']:.1f}% used)")
    
    best_gpu = max(gpu_info, key=lambda x: x['free_gb'])
    
    if best_gpu['free_gb'] < min_free_gb:
        logger.warning(f"No GPU has {min_free_gb}GB free, using GPU {best_gpu['id']}")
    
    print(f"\n✓ Selected GPU {best_gpu['id']} with {best_gpu['free_gb']:.1f}GB free")
    torch.cuda.set_device(best_gpu['id'])
    
    return best_gpu['id']


def setup_gpu(device_id: Optional[int] = None, min_free_gb: float = 15.0):
    """Setup GPU with optimal settings."""
    gc.collect()
    torch.cuda.empty_cache()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Check if CUDA_VISIBLE_DEVICES is set
    import os
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        # When CUDA_VISIBLE_DEVICES is set, PyTorch re-indexes devices starting from 0
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if len(visible_devices) == 1:
            # Only one device is visible, use device 0
            device_id = 0
            print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} - using PyTorch device 0")
        elif device_id is not None and device_id >= len(visible_devices):
            print(f"Warning: device_id {device_id} exceeds visible devices, using device 0")
            device_id = 0
    else:
        if device_id is None:
            device_id = select_best_gpu(min_free_gb)
        else:
            torch.cuda.set_device(device_id)
            print(f"Using specified GPU {device_id}")
    
    device = torch.device(f'cuda:{device_id}')
    
    print(f"=== GPU Setup Complete ===")
    print(f"GPU: {torch.cuda.get_device_name(device_id)}")
    print(f"Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    print(f"Current usage: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    return device


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def overlay_attention_enhanced(
    image: Image.Image,
    attention_weights: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> Image.Image:
    """Create enhanced overlay visualization."""
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Resize attention to match image
    attention_resized = np.array(Image.fromarray(
        (attention_weights * 255).astype(np.uint8)
    ).resize(image.size, Image.BILINEAR))
    
    # Normalize
    attention_resized = attention_resized / attention_resized.max()
    
    # Apply smoothing
    attention_smoothed = gaussian_filter(attention_resized, sigma=1)
    
    # Create heatmap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attention_smoothed)
    heatmap_img = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
    
    # Blend with original
    blended = Image.blend(image.convert('RGB'), heatmap_img, alpha)
    
    return blended


if __name__ == "__main__":
    print("Enhanced MedGemma module loaded successfully")