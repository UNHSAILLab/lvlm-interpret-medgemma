#!/usr/bin/env python3
"""
Single-sample MedGemma attention extraction

Loads the default MIMIC CSV and image base path used by the repository,
runs one generate pass, then extracts a token-conditioned attention map
with robust fallbacks. Saves an overlay PNG and the raw grid (npy).

Usage (defaults pick the first valid sample):
  python scripts/analysis/medgemma_extract_single.py \
      --idx 0 \
      --out-dir results/single_attention_extract \
      --targets pleural effusion

Notes
- The attention map is with respect to the generated answer tokens
  (decoder queries), gated by prompt tokens matching the given targets.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd


def _add_repo_root_to_syspath():
    # Ensure we can import local modules (models/medgemma/..)
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root_to_syspath()

DEFAULT_CSV = None
DEFAULT_IMAGE_BASE = None


def pick_targets(question: str, user_targets: list[str] | None) -> list[str]:
    if user_targets:
        return user_targets
    # Simple heuristic list with a few common medical terms
    terms = [
        "effusion", "pleural", "pneumothorax", "edema", "consolidation",
        "atelectasis", "opacity", "cardiomegaly", "nodule", "mass",
    ]
    q = question.lower()
    chosen = [t for t in terms if t in q]
    # Fallback: pick up to 2 non-trivial words from question
    if not chosen:
        tokens = [w.strip(".,?;:!()[]{}\"'") for w in q.split()]
        chosen = [w for w in tokens if len(w) >= 5][:2]
    return chosen[:3]


def resolve_image_path(row: pd.Series, image_base: Path) -> Path | None:
    # Prefer "ImagePath" column
    if "ImagePath" in row and isinstance(row["ImagePath"], str):
        p = image_base / row["ImagePath"]
        if p.exists():
            return p

    # Try list-like "image_path"
    if "image_path" in row and isinstance(row["image_path"], str):
        try:
            lst = ast.literal_eval(row["image_path"])
            if isinstance(lst, list) and lst:
                p = image_base / str(lst[0])
                if p.exists():
                    return p
        except Exception:
            pass

    # Fallback: study_id.jpg
    if "study_id" in row and isinstance(row["study_id"], str):
        p = image_base / f"{row['study_id']}.jpg"
        if p.exists():
            return p

    return None


def find_first_valid_sample(df: pd.DataFrame, image_base: Path, start_idx: int = 0) -> tuple[int, pd.Series, Path]:
    n = len(df)
    for i in range(start_idx, n):
        row = df.iloc[i]
        img_path = resolve_image_path(row, image_base)
        if img_path is not None:
            return i, row, img_path
    raise RuntimeError("No valid sample with existing image found.")


def main():
    ap = argparse.ArgumentParser(description="Single-sample MedGemma attention extraction")
    ap.add_argument("--csv", default=None, help="Path to MIMIC CSV (default: repo constant)")
    ap.add_argument("--image-base", default=None, help="Base folder for images (default: repo constant)")
    ap.add_argument("--idx", type=int, default=0, help="Row index to try first (will advance until file exists)")
    ap.add_argument("--out-dir", default="results/single_attention_extract", help="Output directory")
    ap.add_argument("--targets", nargs="*", default=None, help="Target words for gating (default inferred from question)")
    ap.add_argument("--max-new", type=int, default=20, help="Max new tokens to generate")
    ap.add_argument("--force-gradcam", action="store_true", help="Force Grad-CAM mode")
    ap.add_argument("--gpu", type=int, default=-1, help="CUDA device index to use (PyTorch-visible). -1 leaves visibility unchanged.")
    args = ap.parse_args()

    # Set CUDA visibility BEFORE importing torch/medgemma modules to avoid mismatched indices
    import os
    if args.gpu is not None and args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Import after setting CUDA visibility
    from models.medgemma.medgemma_launch_mimic_fixed import (
        load_model_enhanced,
        run_generate_with_attention_robust,
        extract_token_conditioned_attention_robust,
        overlay_attention_enhanced,
        MIMIC_CSV_PATH as _DEFAULT_CSV,
        MIMIC_IMAGE_BASE_PATH as _DEFAULT_IMAGE_BASE,
    )
    global DEFAULT_CSV, DEFAULT_IMAGE_BASE
    DEFAULT_CSV = _DEFAULT_CSV
    DEFAULT_IMAGE_BASE = _DEFAULT_IMAGE_BASE

    csv_path = Path(args.csv or DEFAULT_CSV)
    image_base = Path(args.image_base or DEFAULT_IMAGE_BASE)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {csv_path}")
    print(f"Image base: {image_base}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    idx, row, img_path = find_first_valid_sample(df, image_base, start_idx=args.idx)
    question = str(row.get("question", "Is there a pleural effusion?"))
    study_id = str(row.get("study_id", f"row_{idx}"))

    targets = pick_targets(question, args.targets)
    print(f"Selected sample index: {idx}")
    print(f"Study ID: {study_id}")
    print(f"Question: {question}")
    print(f"Targets: {targets}")
    print(f"Image: {img_path}")

    # Load model and processor (avoid nvidia-smi indexing; use PyTorch-visible indices)
    try:
        import torch
        if torch.cuda.is_available():
            # After setting CUDA_VISIBLE_DEVICES, the only visible GPU is index 0
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
    except Exception:
        device = None

    model, processor = load_model_enhanced(device=device)
    device = next(model.parameters()).device

    # Prepare image
    img = Image.open(img_path).convert("RGB")

    # Generate answer (decoder path)
    gen = run_generate_with_attention_robust(
        model, processor, img, question, device=device, max_new_tokens=args.max_new
    )
    print("Generated answer:", gen["generated_text"]) 

    # Extract attention grid (token-conditioned)
    grid, tgt_idx, mode = extract_token_conditioned_attention_robust(
        model,
        processor,
        gen,
        targets,
        pil_image=img,
        prompt=question,
        use_gradcam=args.force_gradcam,
    )
    print(f"Extraction mode: {mode}")
    print(f"Target prompt token indices: {tgt_idx}")

    # Save outputs
    grid_path = out_dir / f"grid_{study_id}.npy"
    np.save(grid_path, grid)

    overlay = overlay_attention_enhanced(img, grid, processor, alpha=0.35, debug_align=False)
    overlay_path = out_dir / f"overlay_{study_id}.png"
    overlay.save(overlay_path)

    meta = {
        "study_id": study_id,
        "index": idx,
        "question": question,
        "generated_answer": gen["generated_text"],
        "targets": targets,
        "mode": mode,
        "image_path": str(img_path),
        "csv_path": str(csv_path),
    }
    with open(out_dir / f"meta_{study_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {overlay_path}")
    print(f"Saved: {grid_path}")
    print(f"Saved: {out_dir / f'meta_{study_id}.json'}")
    print("Note: attention is w.r.t. generated answer tokens; targets gate via prompt tokens.")


if __name__ == "__main__":
    main()
