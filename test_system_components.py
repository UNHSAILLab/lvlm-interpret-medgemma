#!/usr/bin/env python3
"""
Quick test script for MedGemma-4b evaluation
"""

import torch
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

# Test imports
try:
    from medgemma_attention_extraction import SystemConfig, UnifiedAttentionExtractor, TokenGating
    print("✓ Attention extraction system imported")
except Exception as e:
    print(f"✗ Error importing attention extraction: {e}")

try:
    from medgemma_robustness_eval import RobustnessEvaluator, ImagePerturbations
    print("✓ Robustness evaluator imported")
except Exception as e:
    print(f"✗ Error importing robustness evaluator: {e}")

try:
    from medgemma_dataset_evaluator import MainEvaluator
    print("✓ Dataset evaluator imported")
except Exception as e:
    print(f"✗ Error importing dataset evaluator: {e}")

try:
    from medgemma4b_runner import MedGemma4BEvaluator, load_medgemma4b
    print("✓ MedGemma4b runner imported")
except Exception as e:
    print(f"✗ Error importing medgemma4b_runner: {e}")


def test_basic_components():
    """Test basic components without model"""
    print("\n" + "="*60)
    print("Testing Basic Components")
    print("="*60)
    
    # Test configuration
    config = SystemConfig(
        model_id="google/medgemma-4b-it",
        io_paths={
            "data_csv": Path("/home/bsada1/mimic_cxr_hundred_vqa/medical-cxr-vqa-questions_sample_hardpositives.csv"),
            "image_root": Path("/home/bsada1/mimic_cxr_hundred_vqa"),
            "out_dir": Path("./test_output")
        }
    )
    print(f"✓ Config created: {config.model_id}")
    
    # Test token gating
    token_gating = TokenGating()
    test_indices = token_gating.find_target_indices(
        None,  # Will trigger fallback
        [1, 2, 3, 4, 5],
        "is there pleural effusion?",
        ["effusion"]
    )
    print(f"✓ Token gating tested: found {len(test_indices)} indices")
    
    # Test image perturbations
    test_img = Image.new('RGB', (512, 512), 'white')
    perturbed = ImagePerturbations.add_gaussian_noise(test_img, sigma=2)
    print(f"✓ Image perturbation tested: {perturbed.size}")
    
    # Check dataset
    csv_path = config.io_paths["data_csv"]
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset found: {len(df)} samples")
        print(f"  Columns: {df.columns.tolist()[:5]}...")
        print(f"  Strategies: {df['strategy'].unique()[:3]}...")
        
        # Check for images
        img_root = config.io_paths["image_root"]
        sample_img = img_root / df.iloc[0]['image_path']
        if sample_img.exists():
            print(f"✓ Sample image found: {sample_img.name}")
        else:
            # Try alternative paths
            dicom_id = df.iloc[0]['dicom_id']
            for ext in ['.jpg', '.png', '.dcm']:
                alt_path = img_root / f"{dicom_id}{ext}"
                if alt_path.exists():
                    print(f"✓ Sample image found (alt): {alt_path.name}")
                    break
    else:
        print(f"✗ Dataset not found at {csv_path}")


def test_with_mock_model():
    """Test with mock model structure"""
    print("\n" + "="*60)
    print("Testing with Mock Model")
    print("="*60)
    
    class MockModel:
        class Config:
            output_attentions = True
            return_dict = True
            class TextConfig:
                num_attention_heads = 8
                num_key_value_heads = 4
            text_config = TextConfig()
            mm_tokens_per_image = 256
        config = Config()
        device = "cpu"
        
        def eval(self):
            return self
    
    class MockProcessor:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))
            def decode(self, ids, skip_special_tokens=True):
                return "no"
        tokenizer = Tokenizer()
        
        def apply_chat_template(self, messages, **kwargs):
            class MockInputs:
                input_ids = torch.randint(0, 100, (1, 50))
                pixel_values = torch.randn(1, 3, 224, 224)
                def to(self, device):
                    return self
            return MockInputs()
    
    model = MockModel()
    processor = MockProcessor()
    
    config = SystemConfig(model_id="mock_model")
    extractor = UnifiedAttentionExtractor(model, processor, config)
    
    print(f"✓ Mock extractor created")
    print(f"  Architecture: H={config.architecture['H']}, K={config.architecture['K']}")
    print(f"  Grid size: {config.architecture['grid_size']}x{config.architecture['grid_size']}")
    
    # Test uniform grid
    uniform = extractor.create_uniform_grid()
    print(f"✓ Uniform grid: shape={uniform.grid.shape}, mode={uniform.mode}")
    
    # Test GQA aggregation
    test_attn = torch.randn(8, 100, 100)  # [H, Q, KV]
    aggregated = extractor.gqa_aggregate(test_attn, is_cross=True)
    print(f"✓ GQA aggregation: {test_attn.shape} -> {aggregated.shape}")


def main():
    """Run all tests"""
    print("="*60)
    print("MedGemma-4b System Test")
    print("="*60)
    
    # Basic component tests
    test_basic_components()
    
    # Mock model tests
    test_with_mock_model()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)
    print("\nTo run full evaluation with real model:")
    print("  python medgemma4b_runner.py")


if __name__ == "__main__":
    main()