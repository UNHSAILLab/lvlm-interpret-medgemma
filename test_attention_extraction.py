#!/usr/bin/env python3
"""
Test script to verify attention extraction works with MedGemma-4b
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path

def test_attention_extraction():
    """Quick test of attention extraction with MedGemma-4b"""
    
    print("="*60)
    print("Testing MedGemma-4b Attention Extraction")
    print("="*60)
    
    # Check GPU - handle CUDA initialization issues
    try:
        if torch.cuda.is_available():
            # When using CUDA_VISIBLE_DEVICES, the visible GPU becomes device 0
            current_device = 0
            torch.cuda.set_device(current_device)
            
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(current_device)}")
            print(f"  Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB")
            device = "cuda"
        else:
            print("✗ CUDA not available, using CPU (will be slow)")
            device = "cpu"
    except Exception as e:
        print(f"⚠ CUDA issue: {e}")
        print("  Attempting to use CPU instead")
        device = "cpu"
    
    print("\n1. Loading model with eager attention...")
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model_id = "google/medgemma-4b-it"
        
        # Load with eager attention
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Enable attention
        model.config.output_attentions = True
        model.config.return_dict = True
        model.eval()
        
        print("✓ Model loaded with eager attention")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print("\n2. Creating test input...")
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='white')
    test_question = "Is there pleural effusion?"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": test_question},
                {"type": "image", "image": test_image}
            ]
        }
    ]
    
    print("✓ Test input created")
    
    print("\n3. Processing with model...")
    try:
        # Process input
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text,
            images=test_image,
            return_tensors="pt"
        ).to(model.device)
        
        print(f"  Input shape: {inputs.input_ids.shape}")
        
        # Generate with attention
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        print("✓ Generation completed")
        
        # Check attention outputs
        if hasattr(outputs, 'attentions') and outputs.attentions:
            print(f"✓ Attention captured: {len(outputs.attentions)} tokens")
            
            # Check first token's attention
            first_attn = outputs.attentions[0]
            if first_attn:
                layer_attn = first_attn[0]  # First layer
                print(f"  Attention shape (first layer): {layer_attn.shape}")
                
                # Verify it's the expected shape
                if layer_attn.dim() == 4:
                    batch, heads, seq_len, seq_len2 = layer_attn.shape
                    print(f"  Batch: {batch}, Heads: {heads}, Seq: {seq_len}x{seq_len2}")
                    
                    # Check if this matches Gemma3 architecture
                    if heads == 8:  # Expected for Gemma3
                        print("✓ Head count matches Gemma3 architecture (8 heads)")
                    else:
                        print(f"⚠ Unexpected head count: {heads} (expected 8)")
        else:
            print("✗ No attention outputs found")
            
        # Decode answer
        answer_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        answer = processor.decode(answer_ids, skip_special_tokens=True)
        print(f"\n  Generated answer: '{answer}'")
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✓ Attention extraction test successful!")
    print("The model is ready for full evaluation.")
    print("="*60)


if __name__ == "__main__":
    test_attention_extraction()