#!/usr/bin/env python3
"""
Diagnostic script to understand MedGemma's attention mechanism
and find layers that can provide useful attention weights.
"""

import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

def main():
    # Set GPU
    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    else:
        gpu = "5"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"Using GPU {gpu}")
    
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import numpy as np
    
    # Load model and processor
    print("\n=== Loading Model ===")
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=True
    )
    model.eval()
    print("Model loaded")
    
    # Create a dummy image
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "is there pleural effusion?"},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")
    
    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    
    # Examine model structure
    print("\n=== Model Structure ===")
    print(f"Model type: {type(model)}")
    print(f"Config: {model.config.architectures}")
    
    # List all modules
    print("\n=== Key Modules ===")
    for name, module in model.named_modules():
        if any(key in name for key in ['vision', 'cross', 'encoder', 'decoder', 'attention']):
            print(f"  {name}: {type(module).__name__}")
    
    # Try a forward pass
    print("\n=== Forward Pass ===")
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.use_cache = False
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    print(f"Output keys: {outputs.keys()}")
    
    # Check attention outputs
    if hasattr(outputs, 'attentions') and outputs.attentions:
        print(f"Self-attentions available: {len(outputs.attentions)} layers")
        print(f"First attention shape: {outputs.attentions[0].shape}")
    else:
        print("No self-attentions available")
    
    if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
        print(f"Cross-attentions available: {len(outputs.cross_attentions)} layers")
        print(f"First cross-attention shape: {outputs.cross_attentions[0].shape}")
    else:
        print("No cross-attentions available")
    
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        print(f"Hidden states available: {len(outputs.hidden_states)} layers")
        print(f"First hidden state shape: {outputs.hidden_states[0].shape}")
    
    # Try generation with attention
    print("\n=== Generation Test ===")
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False
        )
    
    print(f"Generation output keys: {gen_outputs.keys()}")
    
    # Check if we have attention weights from generation
    if hasattr(gen_outputs, 'attentions') and gen_outputs.attentions:
        print(f"Generation attentions: {type(gen_outputs.attentions)}")
        if isinstance(gen_outputs.attentions, (list, tuple)) and len(gen_outputs.attentions) > 0:
            print(f"Number of generation steps: {len(gen_outputs.attentions)}")
            if gen_outputs.attentions[0]:
                print(f"Attention shape at step 0: {gen_outputs.attentions[0][0].shape}")
    
    # Try to find vision encoder
    print("\n=== Vision Encoder Search ===")
    vision_encoder = None
    for name, module in model.named_modules():
        if 'vision' in name and 'encoder' in name:
            vision_encoder = module
            print(f"Found vision encoder: {name}")
            break
    
    if vision_encoder:
        # Try to hook into vision encoder
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output)
            print(f"  Captured activation shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # Register hook on last layer
        if hasattr(vision_encoder, 'layers'):
            last_layer = list(vision_encoder.layers)[-1]
            hook = last_layer.register_forward_hook(hook_fn)
            
            print("Running forward pass with vision encoder hook...")
            with torch.no_grad():
                _ = model(**inputs)
            
            hook.remove()
            
            if activations:
                print(f"Successfully captured {len(activations)} activations")
            else:
                print("No activations captured")
    
    # Check for image tokens
    print("\n=== Image Token Analysis ===")
    if hasattr(processor, 'image_token'):
        print(f"Image token: {processor.image_token}")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        print(f"Image token ID: {image_token_id}")
        
        # Count image tokens in input
        num_image_tokens = (inputs['input_ids'] == image_token_id).sum().item()
        print(f"Number of image tokens in input: {num_image_tokens}")
    
    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    main()