#!/usr/bin/env python3
"""
FIXED Complete MedGemma Analysis Pipeline for MIMIC-CXR Dataset
Now with improved attention visualization
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import stats
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import logging
import gc
import ast
from dataclasses import dataclass
from collections import Counter
import warnings
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Global configuration for the pipeline"""
    model_id: str = 'google/medgemma-4b-it'
    max_new_tokens: int = 20
    do_sample: bool = False
    max_attention_tokens: int = 3
    attention_layers_to_save: List[int] = None
    batch_size: int = 1
    clear_cache_frequency: int = 2
    max_visualizations: int = 10
    figure_dpi: int = 120  # Increased from 100

    def __post_init__(self):
        if self.attention_layers_to_save is None:
            self.attention_layers_to_save = [33]


# ============================================================================
# IMPROVED VISUALIZATION FUNCTIONS
# ============================================================================

def prepare_attn_grid(attn):
    """Prepare attention data as a square grid"""
    attn = np.asarray(attn, dtype=np.float32)
    if attn.ndim == 1:
        n = int(np.sqrt(attn.size))
        attn = attn[:n * n].reshape(n, n)
    return attn


def tight_body_mask(gray):
    """Create a tight mask for the body region, removing borders and annotations"""
    g = gray.astype(np.uint8)
    # Remove annotations and padding
    base = cv2.GaussianBlur(g, (0, 0), 2)
    _, m = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = 255 - m  # Foreground bright on CXR -> invert
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    return m


def model_view_image(processor, pil_image):
    """Get the exact image as the model sees it using proper denormalization"""
    # MedGemma processor needs both text and image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "dummy"},
                {"type": "image", "image": pil_image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    px = inputs["pixel_values"][0]  # [3, H, W] in model space
    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std = torch.tensor(ip.image_std).view(3, 1, 1)

    # Denormalize and convert to grayscale
    arr = (px * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.uint8)
    return Image.fromarray(gray)


def tight_body_mask(gray):
    """Create a tight mask for the body region, removing borders and annotations"""
    g = gray.astype(np.uint8)
    base = cv2.GaussianBlur(g, (0, 0), 2)
    _, m = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = 255 - m  # Invert for chest X-rays
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    return m


def prepare_attn_grid(attn):
    """Prepare attention data as a square grid"""
    a = np.asarray(attn, dtype=np.float32)
    if a.ndim == 1:
        n = int(np.sqrt(a.size))
        a = a[:n * n].reshape(n, n)
    return a


def strip_border_tokens(attn_grid, k=1):
    """Zero out the outer ring of tokens to remove padding artifacts"""
    g = attn_grid.copy()
    g[:k, :] = 0
    g[-k:, :] = 0
    g[:, :k] = 0
    g[:, -k:] = 0
    return g


def overlay_attention(image_path, attn, out_path, processor, alpha=0.35, debug_align=True):
    """
    Optimized attention overlay with mask-first percentile clipping

    Args:
        image_path: Path to the original X-ray image
        attn: Attention weights (can be 1D array or 2D grid)
        out_path: Path to save the visualization
        processor: MedGemma processor for exact model view
        alpha: Transparency for the attention overlay
        debug_align: Use NEAREST interpolation for alignment debugging
    """
    # Get exact model view
    base_img = model_view_image(processor, Image.open(image_path).convert("RGB"))
    base = np.array(base_img)

    # Create tight body mask
    mask = tight_body_mask(base)

    # Prepare and clean attention grid
    attn = strip_border_tokens(prepare_attn_grid(attn), k=1)
    gh, gw = attn.shape
    H, W = base.shape[:2]

    # Resize with appropriate interpolation
    interp = cv2.INTER_NEAREST if debug_align else cv2.INTER_CUBIC
    heat = cv2.resize(attn, (W, H), interpolation=interp)

    # Compute percentiles only on masked region
    sel = mask > 0
    vals = heat[sel]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (heat.min(), heat.max())

    # Clip and normalize
    heat = np.clip(heat, lo, hi)
    heat = (heat - lo) / (hi - lo + 1e-8)

    # Apply mask
    heat *= sel.astype(np.float32)

    # Create visualization
    fig = plt.figure(dpi=120)
    plt.imshow(base, cmap="gray", vmin=0, vmax=255)
    plt.imshow(heat, alpha=alpha, cmap="jet")

    # Add exact grid lines
    ys = np.linspace(0, H, gh + 1)
    xs = np.linspace(0, W, gw + 1)
    for y in ys:
        plt.axhline(y, color='white', linewidth=0.4, alpha=0.5)
    for x in xs:
        plt.axvline(x, color='white', linewidth=0.4, alpha=0.5)

    plt.axis("off")
    plt.colorbar(fraction=0.025, pad=0.01)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================================
# GPU SELECTION
# ============================================================================

def get_gpu_memory_from_nvidia_smi():
    """Get actual GPU memory usage using nvidia-smi"""
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
    """Select GPU with most free memory"""
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
        for gpu in gpu_info:
            if gpu['free_gb'] >= min_free_gb:
                best_gpu = gpu
                break
        else:
            raise RuntimeError(f"No GPU has at least {min_free_gb}GB free memory")

    print(f"\n✓ Selected GPU {best_gpu['id']} with {best_gpu['free_gb']:.1f}GB free")
    torch.cuda.set_device(best_gpu['id'])

    return best_gpu['id']


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class MedGemmaAnalyzer:
    """Main class for MedGemma analysis"""

    def __init__(self, config: Config, device_id: Optional[int] = None):
        self.config = config

        if device_id is None:
            device_id = select_best_gpu()

        self.device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)

        print(f"\nInitializing MedGemma on cuda:{device_id}...")

        torch.cuda.empty_cache()
        gc.collect()

        self._load_model(device_id)

        self.results = []
        self.attention_cache = {}
        self.samples_processed = 0

    def _load_model(self, device_id: int):
        """Load MedGemma model"""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        print(f"Loading model on cuda:{device_id}...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': device_id},
            attn_implementation="eager",
            tie_word_embeddings=False,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print("✓ Model loaded successfully")
        self._print_memory_usage()

    def _print_memory_usage(self):
        """Print current GPU memory usage"""
        device_id = self.device.index if self.device.index is not None else 0
        allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        print(f"GPU {device_id} Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def process_sample(self, image_path: str, question: str, options: List[str],
                       study_id: str, correct_answer: str) -> Dict:
        """Process a single sample"""

        try:
            image = Image.open(image_path).convert('RGB')

            valid_options = [opt for opt in options if opt]
            prompt = f"""Question: {question}
Answer with 'yes' or 'no'."""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            inputs_gpu = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_gpu,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            answer = self._extract_answer(generated_text)
            attention_data = self._extract_minimal_attention(outputs, inputs_gpu)

            del outputs
            del inputs_gpu
            del inputs
            torch.cuda.empty_cache()

            self.samples_processed += 1

            return {
                'study_id': study_id,
                'question': question,
                'correct_answer': correct_answer,
                'generated_answer': answer,
                'generated_text': generated_text[:100],
                'is_correct': answer == correct_answer,
                'attention_data': attention_data,
                'image_path': image_path
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM for {study_id}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return None

        except Exception as e:
            logger.error(f"Error processing {study_id}: {e}")
            return None

    def _extract_answer(self, text: str) -> str:
        """Extract yes/no answer"""
        text_lower = text.lower()[:50]

        if 'yes' in text_lower[:10]:
            return 'yes'
        elif 'no' in text_lower[:10]:
            return 'no'
        else:
            return 'uncertain'

    def _extract_minimal_attention(self, outputs, inputs) -> Dict:
        """Extract attention data with improved format for visualization"""
        data = {}
        try:
            full_ids = outputs.sequences[:, :-1].to(self.device)
            full_mask = torch.ones_like(full_ids, device=self.device)

            with torch.no_grad():
                attn_out = self.model(
                    input_ids=full_ids,
                    pixel_values=inputs["pixel_values"],
                    attention_mask=full_mask,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

            def summarize(vec: torch.Tensor) -> Dict:
                vec = vec.float()

                # Ensure we only have a perfect square of vision tokens
                g = int(np.sqrt(vec.shape[0]))
                vec = vec[:g * g]  # Keep only perfect square

                vec = vec / (vec.sum() + 1e-8)

                # Keep the raw attention vector for visualization
                raw_attention = vec.cpu().numpy()

                # Create grid
                grid = vec.view(g, g).cpu().numpy()

                h, w = grid.shape
                quads = {
                    "upper_left": grid[:h // 2, :w // 2].mean(),
                    "upper_right": grid[:h // 2, w // 2:].mean(),
                    "lower_left": grid[h // 2:, :w // 2].mean(),
                    "lower_right": grid[h // 2:, w // 2:].mean(),
                }

                return {
                    "regional_focus": max(quads, key=quads.get),
                    "attention_entropy": float(stats.entropy(grid.flatten() + 1e-10)),
                    "attention_grid": grid.tolist(),
                    "raw_attention": raw_attention.tolist(),
                }

            # Use cross-attention (cleaner for vision-language models)
            xattn = getattr(attn_out, "cross_attentions", None)
            if xattn:
                last = xattn[-1][0].mean(0)  # [q_len, kv_len] - average across heads

                # Average last 5 queries for maximum stability
                if last.shape[0] >= 5:
                    vec = last[-5:].mean(0)  # Average last 5 decode steps
                elif last.shape[0] >= 3:
                    vec = last[-3:].mean(0)  # Average last 3 if less than 5
                else:
                    vec = last[-1]  # Just the last query

                return summarize(vec)

            # Fallback to self-attention with image token positions
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is None and hasattr(self.processor.tokenizer, "image_token_id"):
                image_token_id = self.processor.tokenizer.image_token_id

            if attn_out.attentions and image_token_id is not None:
                img_pos = (full_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
                if img_pos.numel() > 0:
                    last = attn_out.attentions[-1][0].mean(0)  # [q_len, key_len]

                    # Average last 5 queries for stability
                    if last.shape[0] >= 5:
                        vec = last[-5:].mean(0)[img_pos]
                    elif last.shape[0] >= 3:
                        vec = last[-3:].mean(0)[img_pos]
                    else:
                        vec = last[-1, img_pos]

                    return summarize(vec)

            return {}
        except Exception as e:
            logger.warning(f"Failed to extract attention: {e}")
            return {}


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_analysis(csv_path: str, image_base_path: str, output_dir: str,
                          sample_size: Optional[int] = None, visualize: bool = True):
    """Run the complete MedGemma analysis pipeline"""

    print("=" * 60)
    print("MEDGEMMA MIMIC-CXR ANALYSIS")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_dir = output_path / "visualizations"
    if visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    config = Config()

    print("\n1. Loading MIMIC-CXR data...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} samples")

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
        print(f"   Using {len(df)} samples for analysis")

    print("\n2. Initializing MedGemma analyzer...")

    try:
        analyzer = MedGemmaAnalyzer(config)
    except RuntimeError as e:
        print(f"\n❌ GPU Selection Failed: {e}")
        print("\nTrying alternative: Manually selecting GPU 1...")
        analyzer = MedGemmaAnalyzer(config, device_id=1)

    print(f"\n3. Processing {len(df)} samples...")
    results = []
    failed_samples = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = Path(image_base_path) / row['ImagePath']

        if not image_path.exists():
            image_path = Path(image_base_path) / f"{row['study_id']}.jpg"
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                failed_samples.append(row['study_id'])
                continue

        if isinstance(row['options'], str):
            try:
                options = ast.literal_eval(row['options'])
            except:
                options = ['yes', 'no']
        else:
            options = row['options'] if row['options'] else ['yes', 'no']

        result = analyzer.process_sample(
            str(image_path),
            row['question'],
            options,
            row['study_id'],
            row['correct_answer']
        )

        if result:
            results.append(result)

            # Use improved visualization
            if visualize and len(results) <= config.max_visualizations:
                if result["attention_data"]:
                    # Try to use raw attention first, fall back to grid
                    attn_to_viz = result["attention_data"].get("raw_attention") or \
                                  result["attention_data"].get("attention_grid")

                    if attn_to_viz:
                        out_file = viz_dir / f"{result['study_id']}.png"

                        # Call overlay_attention
                        overlay_attention(
                            result["image_path"],
                            attn_to_viz,
                            out_file,
                            processor=analyzer.processor,
                            alpha=0.35,
                            debug_align=True  # Set to False after alignment is verified
                        )
        else:
            failed_samples.append(row['study_id'])

        if (idx + 1) % config.clear_cache_frequency == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not results:
        print("\n❌ No samples were successfully processed!")
        return None

    results_df = pd.DataFrame(results)

    print("\n4. Saving results...")
    results_df.to_csv(output_path / 'results.csv', index=False)

    attention_data = {}
    for _, row in results_df.iterrows():
        if row['attention_data']:
            attention_data[row['study_id']] = row['attention_data']

    with open(output_path / 'attention_data.json', 'w') as f:
        json.dump(attention_data, f, indent=2)

    if failed_samples:
        with open(output_path / 'failed_samples.txt', 'w') as f:
            f.write('\n'.join(failed_samples))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(results)}/{len(df)} samples")
    print(f"Failed: {len(failed_samples)} samples")

    if len(results) > 0:
        accuracy = results_df['is_correct'].mean()
        print(f"Overall Accuracy: {accuracy:.2%}")

        regional_focus = []
        for _, row in results_df.iterrows():
            if row['attention_data'] and 'regional_focus' in row['attention_data']:
                focus = row['attention_data']['regional_focus']
                if isinstance(focus, str):
                    regional_focus.append(focus)
                elif isinstance(focus, list):
                    regional_focus.extend(focus)

        if regional_focus:
            from collections import Counter
            region_counts = Counter(regional_focus)
            print("\nRegional Focus Distribution:")
            for region, count in region_counts.most_common():
                print(f"  {region}: {count / len(regional_focus) * 100:.1f}%")

        entropies = []
        for _, row in results_df.iterrows():
            if row['attention_data'] and 'attention_entropy' in row['attention_data']:
                entropies.append(row['attention_data']['attention_entropy'])

        if entropies:
            print(f"\nAttention Entropy:")
            print(f"  Mean: {np.mean(entropies):.2f}")
            print(f"  Std: {np.std(entropies):.2f}")

    print(f"\nResults saved to: {output_path}")

    return results_df


def check_and_free_gpu_memory():
    """Check GPU memory and suggest how to free it"""
    print("\n=== GPU Memory Check ===")

    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)

        if "2843597" in result.stdout:
            print("\n⚠️ WARNING: Process 2843597 is using 66GB of GPU memory!")
            print("To free this memory, run:")
            print("  kill -9 2843597")
            print("\nOr use a different GPU by setting device_id=1 in the code")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not run nvidia-smi")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    check_and_free_gpu_memory()

    CSV_PATH = "/home/bsada1/lvlm-interpret-medgemma/one-pixel-attack/mimic_adapted_questions.csv"
    IMAGE_BASE_PATH = "/home/bsada1/mimic_cxr_hundred_vqa"
    OUTPUT_DIR = "mimic_medgemma_analysis"

    try:
        results = run_complete_analysis(
            csv_path=CSV_PATH,
            image_base_path=IMAGE_BASE_PATH,
            output_dir=OUTPUT_DIR,
            sample_size=10,
            visualize=True
        )

        if results is not None:
            print("\n✅ Success! Analysis completed.")
        else:
            print("\n❌ Analysis failed. Check the error messages above.")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()