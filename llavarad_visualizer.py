#!/usr/bin/env python3
"""
LLaVA-Rad Attention Visualizer and Analysis Platform
Microsoft's LLaVA-Rad model for medical imaging
Developed by SAIL Lab - University of New Haven
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, List, Tuple, Any
import logging
import warnings
from pathlib import Path
import json
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LLaVARadVisualizer:
    """
    Comprehensive attention visualization for Microsoft's LLaVA-Rad model
    Adapted from MedGemma visualizer for consistency
    """
    
    def __init__(self, device=None):
        """Initialize LLaVA-Rad visualizer"""
        self.model = None
        self.processor = None
        self.device = device if device else self.setup_gpu()
        logger.info(f"LLaVA-Rad Visualizer initialized on {self.device}")
    
    def setup_gpu(self, min_free_gb: float = 15.0) -> str:
        """Setup GPU with sufficient memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu"
        
        # Check all available GPUs
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - cached_memory
            free_gb = free_memory / (1024**3)
            
            logger.info(f"GPU {i}: {free_gb:.1f}GB free / {total_memory/(1024**3):.1f}GB total")
            
            if free_gb > max_free_memory and free_gb >= min_free_gb:
                max_free_memory = free_gb
                best_gpu = i
        
        if max_free_memory < min_free_gb:
            raise RuntimeError(f"No GPU with {min_free_gb}GB+ free memory available")
        
        device = f"cuda:{best_gpu}"
        torch.cuda.set_device(best_gpu)
        logger.info(f"Selected GPU {best_gpu} with {max_free_memory:.1f}GB free")
        
        return device
    
    def load_model(self, model_id="Chenglin/llava-med-v1.5-mistral-7b", load_in_8bit=False, load_in_4bit=False):
        """Load LLaVA medical model and processor
        
        Note: Using LLaVA-Med as LLaVA-Rad requires special setup.
        Alternative medical VLMs:
        - Chenglin/llava-med-v1.5-mistral-7b (LLaVA-Med)
        - microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
        """
        logger.info(f"Loading LLaVA medical model: {model_id}")
        
        try:
            # For LLaVA-Med, we need to use the LLaVA loader
            from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoImageProcessor
            
            # Configure quantization if requested
            bnb_config = None
            if load_in_8bit or load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                    bnb_4bit_use_double_quant=load_in_4bit
                )
            
            # Load tokenizer and image processor
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Try to load image processor, fallback to generic one if needed
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            except:
                # Use a compatible image processor
                self.image_processor = AutoImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            
            logger.info("✓ Tokenizer and image processor loaded")
            
            # Create a processor-like interface
            class ProcessorWrapper:
                def __init__(self, tokenizer, image_processor):
                    self.tokenizer = tokenizer
                    self.image_processor = image_processor
                
                def __call__(self, text=None, images=None, **kwargs):
                    encoding = {}
                    if text:
                        text_inputs = self.tokenizer(text, return_tensors="pt", **kwargs)
                        encoding.update(text_inputs)
                    if images is not None:
                        image_inputs = self.image_processor(images, return_tensors="pt")
                        encoding.update(image_inputs)
                    return encoding
                
                def decode(self, *args, **kwargs):
                    return self.tokenizer.decode(*args, **kwargs)
            
            self.processor = ProcessorWrapper(self.tokenizer, self.image_processor)
            
            # Load model with automatic device placement
            model_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            elif self.device != "cpu":
                model_kwargs["device_map"] = {"": self.device}
            
            # Add attention implementation for all models
            model_kwargs["attn_implementation"] = "eager"  # Required for attention extraction
            
            # Try LLaVA model first
            try:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_kwargs
                )
            except:
                # Fallback to AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    **model_kwargs
                )
            
            # Enable attention output
            self.model.config.output_attentions = True
            self.model.eval()
            
            logger.info(f"✓ LLaVA medical model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            # Report memory usage
            if self.device != "cpu":
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                logger.info(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            return self.model, self.processor
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA medical model: {e}")
            # Try alternative approach
            logger.info("Attempting alternative loading method...")
            return self.load_alternative_model()
    
    def load_alternative_model(self):
        """Load alternative vision-language model (llava-hf version)"""
        model_id = "llava-hf/llava-1.5-7b-hf"
        logger.info(f"Loading alternative model: {model_id}")
        
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Load model with eager attention for attention extraction
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": self.device} if self.device != "cpu" else None,
            attn_implementation="eager"  # Required for attention extraction
        )
        
        self.model.config.output_attentions = True
        self.model.eval()
        
        logger.info(f"✓ Alternative model loaded: {model_id}")
        return self.model, self.processor
    
    def extract_attention_weights(self, outputs, layer_indices=None):
        """Extract attention weights from model outputs"""
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            logger.warning("No attention weights in outputs")
            return None
        
        attentions = outputs.attentions
        
        # Default to middle and late layers for LLaVA
        if layer_indices is None:
            num_layers = len(attentions)
            # Sample evenly across layers
            layer_indices = [num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            layer_indices = [i for i in layer_indices if i < num_layers]
        
        attention_maps = []
        for layer_idx in layer_indices:
            if layer_idx < len(attentions):
                # Get attention for this layer
                layer_attention = attentions[layer_idx]
                
                if isinstance(layer_attention, tuple):
                    layer_attention = layer_attention[0]
                
                # Average across heads and batch
                if layer_attention.dim() == 4:  # [batch, heads, seq, seq]
                    avg_attention = layer_attention.mean(dim=[0, 1])
                elif layer_attention.dim() == 3:  # [heads, seq, seq]
                    avg_attention = layer_attention.mean(dim=0)
                else:
                    avg_attention = layer_attention
                
                attention_maps.append(avg_attention.cpu().numpy())
        
        if not attention_maps:
            return None
        
        # Average across selected layers
        combined_attention = np.mean(attention_maps, axis=0)
        return combined_attention
    
    def extract_visual_attention(self, attention_matrix, image_size=(336, 336), patch_size=14):
        """Extract visual attention from full attention matrix"""
        if attention_matrix is None:
            return None
        
        # LLaVA typically uses 24x24 patches for 336x336 images
        num_patches_side = image_size[0] // patch_size
        num_visual_tokens = num_patches_side * num_patches_side
        
        # Identify visual tokens (usually after initial text tokens)
        # This requires understanding the specific tokenization
        # For now, we'll extract a square region that likely contains visual tokens
        seq_len = attention_matrix.shape[0]
        
        # Heuristic: visual tokens are usually in the middle of the sequence
        # Adjust based on actual LLaVA-Rad tokenization
        if seq_len > num_visual_tokens:
            # Try to find the visual token region
            start_idx = max(0, (seq_len - num_visual_tokens) // 2)
            end_idx = min(seq_len, start_idx + num_visual_tokens)
            
            # Extract visual attention
            visual_attention = attention_matrix[start_idx:end_idx, start_idx:end_idx]
            
            # Reshape to grid if possible
            actual_tokens = visual_attention.shape[0]
            grid_size = int(np.sqrt(actual_tokens))
            
            if grid_size * grid_size == actual_tokens:
                visual_attention = visual_attention.mean(axis=1)  # Average across sequence
                visual_attention = visual_attention.reshape(grid_size, grid_size)
            else:
                # Fallback: resize to expected grid
                visual_attention = visual_attention.mean(axis=1)
                # Pad or trim to fit square grid
                target_size = num_patches_side * num_patches_side
                if len(visual_attention) < target_size:
                    visual_attention = np.pad(visual_attention, 
                                             (0, target_size - len(visual_attention)),
                                             mode='constant')
                else:
                    visual_attention = visual_attention[:target_size]
                visual_attention = visual_attention.reshape(num_patches_side, num_patches_side)
            
            return visual_attention
        
        return None
    
    def generate_with_attention(self, image, question, max_new_tokens=100):
        """Generate answer with attention tracking"""
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare conversation for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}\nAnswer with only 'yes' or 'no'."},
                    {"type": "image"},
                ],
            },
        ]
        
        # Apply chat template and process
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process image and text
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in inputs.items()}
        
        # Generate with attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode answer
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        answer = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Extract attention
        attention_weights = None
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # LLaVA generate returns attentions per generation step
            # We'll focus on the first generation step
            first_step_attentions = outputs.attentions[0] if outputs.attentions else None
            if first_step_attentions:
                attention_weights = self.extract_attention_weights(
                    type('', (), {'attentions': first_step_attentions})()
                )
        
        # Extract visual attention
        visual_attention = None
        if attention_weights is not None:
            visual_attention = self.extract_visual_attention(attention_weights)
        
        return {
            'answer': answer,
            'attention_weights': attention_weights,
            'visual_attention': visual_attention,
            'outputs': outputs
        }
    
    def visualize_attention(self, image, visual_attention, question=None, answer=None, 
                          save_path=None, show_plot=True):
        """Visualize attention on image"""
        if visual_attention is None:
            logger.warning("No visual attention to visualize")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(visual_attention, cmap='hot', interpolation='nearest')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(image)
        
        # Resize attention to match image size
        attention_resized = np.array(Image.fromarray(
            (visual_attention * 255).astype(np.uint8)
        ).resize(image.size, Image.BILINEAR))
        
        axes[2].imshow(attention_resized, alpha=0.5, cmap='hot')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        # Add question and answer as title
        if question and answer:
            fig.suptitle(f'Q: {question[:100]}...\nA: {answer[:100]}...', 
                        fontsize=10, y=1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def compute_attention_metrics(self, visual_attention):
        """Compute attention quality metrics"""
        if visual_attention is None:
            return {}
        
        # Normalize attention
        attention_norm = visual_attention / (visual_attention.sum() + 1e-10)
        
        # Compute metrics
        metrics = {
            'max_attention': float(visual_attention.max()),
            'mean_attention': float(visual_attention.mean()),
            'std_attention': float(visual_attention.std()),
            'entropy': float(-np.sum(attention_norm * np.log(attention_norm + 1e-10))),
            'sparsity': float((visual_attention > visual_attention.mean()).sum() / visual_attention.size),
            'focus_score': float(visual_attention.max() / (visual_attention.mean() + 1e-10))
        }
        
        # Regional distribution (quarters)
        h, w = visual_attention.shape
        metrics['top_half'] = float(visual_attention[:h//2, :].sum() / visual_attention.sum())
        metrics['bottom_half'] = float(visual_attention[h//2:, :].sum() / visual_attention.sum())
        metrics['left_half'] = float(visual_attention[:, :w//2].sum() / visual_attention.sum())
        metrics['right_half'] = float(visual_attention[:, w//2:].sum() / visual_attention.sum())
        
        return metrics
    
    def analyze_image(self, image_path, question, save_visualizations=False, output_dir=None):
        """Complete analysis pipeline for a single image"""
        logger.info(f"Analyzing: {image_path}")
        logger.info(f"Question: {question}")
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Generate answer with attention
        results = self.generate_with_attention(image, question)
        
        # Compute metrics
        metrics = self.compute_attention_metrics(results['visual_attention'])
        
        # Prepare analysis results
        analysis = {
            'question': question,
            'answer': results['answer'],
            'attention_metrics': metrics,
            'visual_attention': results['visual_attention']
        }
        
        # Visualize if requested
        if save_visualizations and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from question
            safe_question = question[:50].replace('/', '_').replace(' ', '_')
            save_path = output_dir / f"llava_rad_{safe_question}.png"
            
            self.visualize_attention(
                image, 
                results['visual_attention'],
                question=question,
                answer=results['answer'],
                save_path=save_path,
                show_plot=False
            )
        
        return analysis
    
    def extract_answer(self, text):
        """Extract yes/no answer from model output"""
        text = text.lower().strip()
        
        # Direct yes/no at start
        if text.startswith('yes'):
            return 'yes'
        if text.startswith('no'):
            return 'no'
        
        # Common patterns
        no_patterns = [
            'no,', 'no.', 'there is no', 'there are no',
            'does not', "doesn't", 'absent', 'not present',
            'not visible', 'cannot be seen', 'not observed',
            'negative for', 'without', 'normal'
        ]
        
        yes_patterns = [
            'yes,', 'yes.', 'there is', 'there are',
            'present', 'visible', 'observed', 'seen',
            'positive for', 'shows', 'demonstrates',
            'evident', 'apparent', 'suggests', 'indicates'
        ]
        
        # Check patterns
        for pattern in no_patterns:
            if pattern in text[:100]:  # Check first 100 chars
                return 'no'
        
        for pattern in yes_patterns:
            if pattern in text[:100]:
                return 'yes'
        
        # Fallback
        if 'yes' in text[:50]:
            return 'yes'
        if 'no' in text[:50]:
            return 'no'
        
        return 'uncertain'

def setup_llava_rad_enhanced(device=None):
    """Setup LLaVA-Rad with enhanced attention extraction"""
    visualizer = LLaVARadVisualizer(device=device)
    
    # Load model with 8-bit quantization for memory efficiency
    model, processor = visualizer.load_model(load_in_8bit=True)
    
    return visualizer, model, processor

def main():
    """Test LLaVA-Rad visualizer"""
    print("="*80)
    print("LLaVA-Rad Vision-Language Model - Attention Visualizer")
    print("Microsoft's Medical Vision-Language Model")
    print("="*80)
    
    # Setup
    visualizer = LLaVARadVisualizer()
    
    # Load model
    visualizer.load_model(load_in_8bit=True)
    
    # Test with sample image
    test_image = "/home/bsada1/mimic_cxr_hundred_vqa/p10_p10000032_s50414267_2373b6a3-f5121edd-63dc44ac-0c4e33e8-ae2d83d7.jpg"
    test_question = "Is there cardiomegaly?"
    
    if os.path.exists(test_image):
        analysis = visualizer.analyze_image(
            test_image,
            test_question,
            save_visualizations=True,
            output_dir="llava_rad_visualizations"
        )
        
        print(f"\nQuestion: {analysis['question']}")
        print(f"Answer: {analysis['answer']}")
        print(f"Extracted: {visualizer.extract_answer(analysis['answer'])}")
        print("\nAttention Metrics:")
        for metric, value in analysis['attention_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print(f"Test image not found: {test_image}")
    
    # Clean up
    torch.cuda.empty_cache()
    print("\n✓ LLaVA-Rad visualizer ready for use")

if __name__ == "__main__":
    main()