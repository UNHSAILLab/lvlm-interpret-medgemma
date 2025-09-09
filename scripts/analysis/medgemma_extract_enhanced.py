#!/usr/bin/env python3
"""
Enhanced MedGemma Attention Extraction CLI
==========================================
Single-sample extraction with multiple attention modes and robust fallbacks.

Usage:
    python scripts/analysis/medgemma_extract_enhanced.py \
        --idx 0 \
        --mode auto \
        --out-dir results/enhanced_attention \
        --targets "pleural" "effusion" \
        --max-new 16 \
        --gpu 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import os

import numpy as np
import pandas as pd
from PIL import Image


def setup_environment(gpu: Optional[int] = None):
    """Setup CUDA environment before importing torch."""
    if gpu is not None and gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"CUDA_VISIBLE_DEVICES set to {gpu}")


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Enhanced MedGemma attention extraction with multiple modes"
    )
    
    # Data arguments
    ap.add_argument("--csv", type=str, default=None,
                   help="Path to MIMIC CSV file (default: use repo default)")
    ap.add_argument("--image-base", type=str, default=None,
                   help="Base path for images (default: use repo default)")
    ap.add_argument("--idx", type=int, default=0,
                   help="Sample index to process")
    
    # Attention extraction arguments
    ap.add_argument("--mode", type=str, default="auto",
                   choices=["auto", "cross", "self", "gradcam", "uniform"],
                   help="Attention extraction mode (default: auto)")
    ap.add_argument("--targets", nargs="+", type=str, default=None,
                   help="Target words to focus on (default: infer from question)")
    ap.add_argument("--use-group-aware", action="store_true", default=True,
                   help="Use group-aware head aggregation for GQA")
    ap.add_argument("--no-group-aware", dest="use_group_aware", action="store_false",
                   help="Disable group-aware aggregation")
    
    # Model arguments
    ap.add_argument("--model-id", type=str, default="google/medgemma-4b-it",
                   help="Model ID to load")
    ap.add_argument("--use-fast-tokenizer", action="store_true", default=True,
                   help="Use fast tokenizer (enables offset mapping)")
    ap.add_argument("--use-slow-tokenizer", dest="use_fast_tokenizer", 
                   action="store_false",
                   help="Use slow tokenizer (compatibility mode)")
    
    # Generation arguments
    ap.add_argument("--max-new", type=int, default=16,
                   help="Maximum new tokens to generate (default: 16)")
    ap.add_argument("--temperature", type=float, default=0.7,
                   help="Generation temperature")
    
    # Output arguments
    ap.add_argument("--out-dir", type=str, default="results/enhanced_attention",
                   help="Output directory for results")
    ap.add_argument("--alpha", type=float, default=0.5,
                   help="Overlay transparency (0=image, 1=attention)")
    ap.add_argument("--colormap", type=str, default="jet",
                   help="Colormap for attention overlay")
    
    # System arguments
    ap.add_argument("--gpu", type=int, default=None,
                   help="GPU device to use (None=auto-select)")
    ap.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging")
    
    return ap.parse_args()


def infer_targets_from_question(question: str) -> List[str]:
    """Infer target words from question text."""
    
    # Common medical terms to look for
    medical_terms = [
        'effusion', 'pleural', 'pneumothorax', 'consolidation',
        'opacity', 'infiltrate', 'edema', 'cardiomegaly',
        'atelectasis', 'pneumonia', 'fracture', 'nodule',
        'mass', 'lesion', 'abnormality'
    ]
    
    question_lower = question.lower()
    targets = []
    
    for term in medical_terms:
        if term in question_lower:
            targets.append(term)
    
    # Also check for common patterns
    if 'is there' in question_lower or 'any' in question_lower:
        # Extract the condition being asked about
        words = question_lower.split()
        for i, word in enumerate(words):
            if word in ['there', 'any'] and i + 1 < len(words):
                next_word = words[i + 1].rstrip('?.,')
                if next_word not in ['a', 'an', 'the'] and next_word not in targets:
                    targets.append(next_word)
    
    return targets if targets else ['abnormality']  # Default fallback


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup CUDA environment BEFORE importing torch
    setup_environment(args.gpu)
    
    # Now import torch-dependent modules
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from models.medgemma.medgemma_enhanced import (
        load_model_enhanced,
        generate_with_attention_enhanced,
        extract_token_conditioned_attention_enhanced,
        overlay_attention_enhanced,
        MIMIC_CSV_PATH,
        MIMIC_IMAGE_BASE_PATH
    )
    
    # Setup paths
    csv_path = Path(args.csv or MIMIC_CSV_PATH)
    image_base = Path(args.image_base or MIMIC_IMAGE_BASE_PATH)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Enhanced MedGemma Attention Extraction")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Image base: {image_base}")
    print(f"Output dir: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Max new tokens: {args.max_new}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter to samples with images
    df_with_images = df[df['image_path'].notna()].copy()
    
    if len(df_with_images) == 0:
        print("ERROR: No samples with images found")
        return 1
    
    # Select sample
    if args.idx >= len(df_with_images):
        print(f"ERROR: Index {args.idx} out of range (max: {len(df_with_images)-1})")
        return 1
    
    sample = df_with_images.iloc[args.idx]
    
    # Get image path
    image_filename = sample['image_path']
    
    # Handle if image_path is parsed as a list string
    if isinstance(image_filename, str) and image_filename.startswith('['):
        import ast
        try:
            image_filename = ast.literal_eval(image_filename)[0]
        except:
            image_filename = image_filename.strip("[]'\"")
    
    if '/' in image_filename:
        image_filename = image_filename.split('/')[-1]
    
    image_path = image_base / image_filename
    
    print(f"\n=== Sample {args.idx} ===")
    print(f"Study ID: {image_filename.replace('.jpg', '')}")
    print(f"Question: {sample['question']}")
    print(f"Image: {image_path}")
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1
    
    # Determine targets
    if args.targets:
        targets = args.targets
    else:
        targets = infer_targets_from_question(sample['question'])
    
    print(f"Targets: {targets}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Load model
    print("\n=== Loading Model ===")
    try:
        model, processor = load_model_enhanced(
            model_id=args.model_id,
            use_fast_tokenizer=args.use_fast_tokenizer
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return 1
    
    # Generate answer with attention
    print("\n=== Generating Answer ===")
    try:
        outputs, inputs, answer = generate_with_attention_enhanced(
            model=model,
            processor=processor,
            image=image,
            question=sample['question'],
            max_new_tokens=args.max_new,
            temperature=args.temperature
        )
        
        print(f"Generated answer: {answer}")
        
    except Exception as e:
        print(f"ERROR: Generation failed: {e}")
        return 1
    
    # Extract attention
    print(f"\n=== Extracting Attention (mode: {args.mode}) ===")
    try:
        attention_grid, metadata = extract_token_conditioned_attention_enhanced(
            outputs=outputs,
            inputs=inputs,
            processor=processor,
            target_prompt_indices=None,  # Will be computed internally
            mode=args.mode,
            model=model,
            question_text=sample['question'],
            target_words=targets,
            use_group_aware=args.use_group_aware
        )
        
        print(f"Extraction successful!")
        print(f"Mode used: {metadata['mode_used']}")
        print(f"Grid shape: {metadata['grid_shape']}")
        print(f"Fallback chain: {' -> '.join(metadata['fallback_chain'])}")
        
        if metadata['target_indices']:
            print(f"Target indices: {metadata['target_indices'][:10]}...")
        
        # Log diagnostics if verbose
        if args.verbose:
            print("\n=== Diagnostics ===")
            for key, value in metadata['diagnostics'].items():
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"ERROR: Attention extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create overlay
    print("\n=== Creating Visualization ===")
    try:
        overlay = overlay_attention_enhanced(
            image=image,
            attention_weights=attention_grid,
            alpha=args.alpha,
            colormap=args.colormap
        )
        
        # Save outputs
        study_id = image_filename.replace('.jpg', '')
        
        # Save overlay image
        overlay_path = out_dir / f"overlay_{study_id}_{args.mode}.png"
        overlay.save(overlay_path)
        print(f"Saved: {overlay_path}")
        
        # Save attention grid
        grid_path = out_dir / f"grid_{study_id}_{args.mode}.npy"
        np.save(grid_path, attention_grid)
        print(f"Saved: {grid_path}")
        
        # Save metadata
        metadata['sample'] = {
            'index': args.idx,
            'study_id': study_id,
            'question': sample['question'],
            'answer': answer,
            'targets': targets
        }
        metadata['parameters'] = {
            'mode': args.mode,
            'max_new_tokens': args.max_new,
            'temperature': args.temperature,
            'use_group_aware': args.use_group_aware,
            'use_fast_tokenizer': args.use_fast_tokenizer
        }
        
        meta_path = out_dir / f"meta_{study_id}_{args.mode}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved: {meta_path}")
        
    except Exception as e:
        print(f"ERROR: Visualization failed: {e}")
        return 1
    
    print("\n=== Complete ===")
    print(f"Successfully processed sample {args.idx}")
    print(f"Results saved to {out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())