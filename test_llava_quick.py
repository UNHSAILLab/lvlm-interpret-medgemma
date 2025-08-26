#!/usr/bin/env python3
"""Quick test of LLaVA model for medical VQA"""

import torch
import os
from PIL import Image

# Force GPU 5
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

print("Testing LLaVA model setup...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

from llava_rad_visualizer import LLaVARadVisualizer

# Create visualizer
visualizer = LLaVARadVisualizer()
print(f"Using device: {visualizer.device}")

# Load model
print("\nLoading LLaVA model (this may take a minute)...")
try:
    model, processor = visualizer.load_model(load_in_8bit=True)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Test with sample image
test_image_path = "/home/bsada1/mimic_cxr_hundred_vqa/0009a9fb-eb905e90-824cad7c-16d40468-007f0038.jpg"
test_question = "Is there cardiomegaly?"

if os.path.exists(test_image_path):
    print(f"\nTesting with sample image...")
    print(f"Question: {test_question}")
    
    image = Image.open(test_image_path).convert('RGB')
    
    # Generate answer
    result = visualizer.generate_with_attention(image, test_question)
    
    print(f"Raw answer: {result['answer'][:200]}")
    extracted = visualizer.extract_answer(result['answer'])
    print(f"Extracted answer: {extracted}")
    
    # Check attention
    if result['visual_attention'] is not None:
        print(f"Visual attention shape: {result['visual_attention'].shape}")
        metrics = visualizer.compute_attention_metrics(result['visual_attention'])
        print("Attention metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("No visual attention extracted")
else:
    print(f"Test image not found: {test_image_path}")

# Clean up
torch.cuda.empty_cache()
print("\n✓ LLaVA test complete!")