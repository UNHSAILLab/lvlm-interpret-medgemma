# MedGemma-4b Attention Analysis System

Unified attention extraction and analysis system for MedGemma-4b (Gemma3ForConditionalGeneration) on medical chest X-ray datasets.

## 🚀 Quick Start

```bash
# Set GPU
export CUDA_VISIBLE_DEVICES=5

# Test single sample
python quick_test_single_sample.py

# Run evaluation (20 samples)
python run_medgemma_evaluation_main.py

# Test components
python test_system_components.py
```

## 📁 Project Structure

### Core System
- `medgemma_attention_extraction.py` - Attention extraction with GQA support
- `medgemma_dataset_evaluator.py` - Dataset processing and metrics
- `medgemma_robustness_eval.py` - Robustness testing
- `medgemma4b_runner.py` - MedGemma-4b specific implementation

### Execution Scripts
- `run_medgemma_evaluation_main.py` - Main evaluation runner
- `quick_test_single_sample.py` - Single sample test
- `test_attention_extraction.py` - Attention extraction test
- `test_system_components.py` - Component tests

### Configuration
- `medgemma4b_config.json` - System configuration

### Documentation
- `docs/QUICKSTART_GUIDE.md` - Detailed usage guide
- `docs/SYSTEM_ARCHITECTURE.md` - Technical architecture
- `docs/IMPLEMENTATION_SUMMARY.md` - Implementation details

## 🔧 Key Features

- **GQA-Aware Attention**: Handles Gemma3's grouped query attention (H=8, K=4)
- **Medical Term Gating**: Focuses on relevant medical terms in questions
- **Body Mask Generation**: Anatomical region detection for chest X-rays
- **Comprehensive Metrics**: Entropy, focus, spatial distribution analysis
- **Hard Positive Analysis**: Tracks question variant consistency

## 📊 Output Structure

```
medgemma4b_results_fixed/
├── evaluation_results.csv     # Complete metrics
├── grids/
│   ├── {study_id}_grid.npy   # 16×16 attention arrays
│   └── {study_id}_meta.json  # Metrics and metadata
├── overlays/                  # Attention overlays on X-rays
└── visualizations/            # Multi-panel analysis figures
```

## 🎯 Dataset

Configured for MIMIC-CXR with hard positive question variants:
- Path: `/home/bsada1/mimic_cxr_hundred_vqa/`
- CSV: `medical-cxr-vqa-questions_sample_hardpositives.csv`

## 📈 Architecture Details

**MedGemma-4b (Gemma3ForConditionalGeneration)**:
- Vision: SigLIP, 27 layers, 64×64 patches
- Projector: AvgPool(4,4) → 16×16 = 256 tokens
- Language: 34 layers with GQA
  - 8 attention heads (Q: 2560→2048)
  - 4 key-value heads (KV: 2560→1024)

## 📝 Citation

If using this system, please cite:
```bibtex
@software{medgemma_attention_2024,
  title={MedGemma-4b Attention Analysis System},
  year={2024}
}
```

---
**Status**: ✅ Operational and tested on MIMIC-CXR dataset