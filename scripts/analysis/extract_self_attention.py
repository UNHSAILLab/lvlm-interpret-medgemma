#!/usr/bin/env python3
"""
Extract self-attention from MedGemma focusing on how generated tokens
attend to image tokens.
"""

import sys
import os
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Add repo root
sys.path.insert(0, str(Path(__file__).parents[2]))


def extract_image_attention(outputs, processor, prompt_len=None):
    """
    Extract attention from generated tokens to image tokens.
    
    MedGemma uses self-attention where generated tokens can attend to image tokens.
    Image tokens are typically at positions corresponding to <image_soft_token> IDs.
    """
    
    if not hasattr(outputs, 'attentions') or not outputs.attentions:
        return None, "No attention weights available"
    
    # Get last layer attention (most semantic)
    # Shape: [batch, heads, seq_len, seq_len]
    last_attention = outputs.attentions[-1]
    batch_size, num_heads, seq_len, _ = last_attention.shape
    
    # Find image token positions
    # In MedGemma, image tokens are marked with image_token_id
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    
    # Get the input IDs to find image token positions
    if hasattr(outputs, 'sequences'):
        input_ids = outputs.sequences[0]
    else:
        # Fallback - assume image tokens are in a continuous block
        # Typically after BOS and before text
        image_start = 2  # After BOS
        image_end = image_start + 256  # 256 image tokens
        image_positions = list(range(image_start, image_end))
    
    # If we have actual token IDs, find image positions
    if 'input_ids' in dir(outputs):
        image_positions = (outputs.input_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    
    # Extract attention from last generated tokens to image tokens
    # Focus on the last few tokens (the answer)
    if prompt_len is not None and seq_len > prompt_len:
        # Tokens after the prompt are generated
        generated_start = prompt_len
    else:
        # Assume last 20 tokens are generated
        generated_start = max(0, seq_len - 20)
    
    # Get attention from generated tokens to image tokens
    if len(image_positions) > 0:
        # Average attention across heads and generated tokens
        attn_to_images = last_attention[:, :, generated_start:, image_positions].mean(dim=(0, 1, 2))
        
        # Reshape to 16x16 grid (assuming 256 image tokens)
        if len(attn_to_images) == 256:
            attention_grid = attn_to_images.reshape(16, 16)
        else:
            # Fallback to uniform if unexpected size
            attention_grid = torch.ones(16, 16) / 256
    else:
        # No image tokens found, use uniform
        attention_grid = torch.ones(16, 16) / 256
    
    return attention_grid.cpu().numpy(), "self_attention"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/self_attention")
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU {args.gpu}")
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    # Load model
    print("Loading model...")
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
        output_attentions=True
    )
    model.eval()
    
    # Load data
    csv_path = Path("/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv")
    image_base = Path("/home/bsada1/mimic_cxr_hundred_vqa")
    
    df = pd.read_csv(csv_path)
    sample = df.iloc[args.idx]
    
    # Get image
    image_filename = sample['image_path']
    if isinstance(image_filename, str) and image_filename.startswith('['):
        import ast
        image_filename = ast.literal_eval(image_filename)[0]
    
    image_path = image_base / image_filename
    image = Image.open(image_path).convert('RGB')
    
    print(f"\nProcessing sample {args.idx}:")
    print(f"Question: {sample['question']}")
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample['question']},
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
    
    prompt_len = inputs['input_ids'].shape[1]
    
    # Generate with attention
    print("Generating answer...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False
        )
    
    # Decode answer
    generated_ids = outputs.sequences[0][prompt_len:]
    answer = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"Answer: {answer}")
    
    # Extract attention
    print("Extracting attention...")
    
    # Method 1: Try to get attention from generation outputs
    if hasattr(outputs, 'attentions') and outputs.attentions:
        # Generation returns attention for each step
        # We want the attention from the last few generated tokens
        last_step_attentions = outputs.attentions[-1]  # Last generation step
        if last_step_attentions:
            last_layer = last_step_attentions[-1]  # Last layer
            
            # Find image token positions in the sequence
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            full_sequence = outputs.sequences[0]
            image_positions = (full_sequence == image_token_id).nonzero(as_tuple=False).squeeze(-1)
            
            if len(image_positions) > 0:
                # Extract attention to image tokens
                # Shape: [batch, heads, 1, seq_len] -> we want [:, :, 0, image_positions]
                attn_to_images = last_layer[:, :, 0, image_positions].mean(dim=(0, 1))
                
                # Reshape to grid
                if len(attn_to_images) == 256:
                    # Convert from bfloat16 to float32 before numpy conversion
                    attention_grid = attn_to_images.reshape(16, 16).float().cpu().numpy()
                    mode = "self_attention_generation"
                else:
                    attention_grid = np.ones((16, 16)) / 256
                    mode = "uniform_fallback"
            else:
                attention_grid = np.ones((16, 16)) / 256
                mode = "uniform_no_image_tokens"
    else:
        # Method 2: Do a forward pass with the full generated sequence
        full_ids = outputs.sequences[0].unsqueeze(0)
        with torch.no_grad():
            forward_outputs = model(
                input_ids=full_ids,
                pixel_values=inputs['pixel_values'],
                output_attentions=True,
                return_dict=True
            )
        
        attention_grid, mode = extract_image_attention(forward_outputs, processor, prompt_len)
    
    # Normalize attention
    attention_grid = attention_grid / (attention_grid.max() + 1e-8)
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    study_id = image_filename.replace('.jpg', '')
    
    # Save grid
    np.save(out_dir / f"grid_{study_id}.npy", attention_grid)
    
    # Save metadata
    metadata = {
        "study_id": study_id,
        "index": args.idx,
        "question": sample['question'],
        "answer": answer,
        "mode": mode,
        "attention_shape": list(attention_grid.shape),
        "attention_stats": {
            "min": float(attention_grid.min()),
            "max": float(attention_grid.max()),
            "mean": float(attention_grid.mean()),
            "std": float(attention_grid.std())
        }
    }
    
    with open(out_dir / f"meta_{study_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(attention_grid, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"Attention ({mode})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    attention_resized = np.array(Image.fromarray(
        (attention_grid * 255).astype(np.uint8)
    ).resize(image.size, Image.BILINEAR))
    attention_resized = attention_resized / attention_resized.max()
    
    axes[2].imshow(image)
    axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.suptitle(f"Sample {args.idx}: {sample['question']}")
    plt.tight_layout()
    plt.savefig(out_dir / f"visualization_{study_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {out_dir}")
    print(f"Mode: {mode}")
    print(f"Attention stats: min={attention_grid.min():.4f}, max={attention_grid.max():.4f}, "
          f"mean={attention_grid.mean():.4f}, std={attention_grid.std():.4f}")


if __name__ == "__main__":
    main()