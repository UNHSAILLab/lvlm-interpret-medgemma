#!/usr/bin/env python3
"""
Quick test with a single MIMIC-CXR sample
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import numpy as np

# Import our system
from medgemma_attention_extraction import SystemConfig, UnifiedAttentionExtractor, TokenGating, BodyMaskGenerator, AttentionMetrics
from medgemma4b_runner import load_medgemma4b

def test_single_sample():
    """Test with one real MIMIC-CXR sample"""
    
    print("="*60)
    print("Single Sample Test - MedGemma-4b on MIMIC-CXR")
    print("="*60)
    
    # Load dataset
    csv_path = Path("/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv")
    df = pd.read_csv(csv_path)
    
    # Get first sample
    sample = df.iloc[0]
    print(f"\nSample info:")
    print(f"  Study ID: {sample.study_id}")
    print(f"  Question: {sample.question}")
    print(f"  Answer: {sample.answer}")
    print(f"  Strategy: {sample.strategy}")
    
    # Load image
    img_root = Path("/home/bsada1/mimic_cxr_hundred_vqa")
    img_path = img_root / sample.image_path
    
    if not img_path.exists():
        print(f"✗ Image not found: {img_path}")
        return
        
    image = Image.open(img_path).convert("RGB")
    print(f"✓ Image loaded: {image.size}")
    
    # Load model
    print("\nLoading model...")
    model, processor = load_medgemma4b("cuda")
    print("✓ Model loaded")
    
    # Create config
    config = SystemConfig(
        model_id="google/medgemma-4b-it",
        device="cuda"
    )
    
    # Initialize extractor
    extractor = UnifiedAttentionExtractor(model, processor, config)
    print(f"✓ Extractor initialized")
    print(f"  Architecture: H={config.architecture['H']}, K={config.architecture['K']}")
    
    # Process sample
    print("\nProcessing sample...")
    
    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample.question},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # Process input
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    
    prompt_len = inputs.input_ids.shape[1]
    print(f"  Prompt length: {prompt_len}")
    
    # Generate with attention
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            output_attentions=True,
            return_dict_in_generate=True,
            temperature=0.0,
            do_sample=False
        )
    
    # Extract answer
    answer_ids = outputs.sequences[0][prompt_len:]
    answer = processor.decode(answer_ids, skip_special_tokens=True)
    print(f"  Generated answer: {answer[:100]}...")
    print(f"  Ground truth: {sample.answer}")
    
    # Re-run for attention extraction
    full_ids = outputs.sequences[:, :-1]
    with torch.no_grad():
        attn_outputs = model(
            input_ids=full_ids,
            pixel_values=inputs.get('pixel_values'),
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
    
    print(f"✓ Attention captured")
    
    # Extract attention with our system
    target_terms = ["effusion", "pleural", "fluid"]
    prompt_ids = full_ids[0, :prompt_len].tolist()
    
    gating_indices = TokenGating.find_target_indices(
        processor.tokenizer,
        prompt_ids,
        sample.question,
        target_terms
    )
    print(f"  Gating indices: {len(gating_indices)} tokens")
    
    # Extract attention maps
    attention_result = extractor.extract_attention_maps(
        attn_outputs,
        prompt_len,
        full_ids,
        gating_indices,
        mode_order=["self", "gradcam", "uniform"],
        messages=messages
    )
    
    print(f"✓ Attention extracted: mode={attention_result.mode}")
    print(f"  Grid shape: {attention_result.grid.shape}")
    print(f"  Grid stats: min={attention_result.grid.min():.3f}, max={attention_result.grid.max():.3f}")
    
    # Generate body mask and metrics
    mask = BodyMaskGenerator.create_body_mask(image)
    metrics = AttentionMetrics.compute_metrics(attention_result.grid, mask)
    
    print(f"\n✓ Metrics computed:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.3f}")
    
    # Save visualization
    from medgemma_attention_extraction import Visualizer
    output_path = Path("./test_output")
    output_path.mkdir(exist_ok=True)
    
    Visualizer.save_visualization(
        image,
        attention_result.grid,
        mask,
        output_path / "test_visualization.png",
        title=f"Test: {sample.question}"
    )
    
    print(f"\n✓ Visualization saved to: {output_path / 'test_visualization.png'}")
    
    print("\n" + "="*60)
    print("✓ Single sample test successful!")
    print("="*60)


if __name__ == "__main__":
    test_single_sample()