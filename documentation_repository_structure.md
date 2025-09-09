could # Repository Structure

## lvlm-interpret-medgemma
Medical Vision-Language Model Interpretation and Analysis

### Directory Organization

```
lvlm-interpret-medgemma/
│
├── models/                     # Model implementations
│   ├── medgemma_launch_mimic_fixed.py
│   ├── llava_rad_visualizer.py
│   └── medgemma/
│       └── medgemma_launch_mimic_fixed.py
│
├── analysis/                   # Analysis scripts
│   ├── analyze_all_attention_changes.py
│   ├── analyze_attention_llava_rad.py
│   ├── analyze_llava_matched_to_medgemma.py
│   ├── batch_analysis_100samples.py
│   └── faithfulness_validation.py
│
├── results/                    # All analysis results
│   ├── medgemma_attention_analysis/
│   ├── llava_rad_attention_analysis/
│   ├── llava_matched_attention_analysis/
│   ├── question_phrasing_results/
│   ├── batch_analysis_results/
│   └── llava_100samples_results/
│
├── scripts/                    # Utility scripts
│   ├── analysis/
│   │   ├── llava_batch_analysis.py
│   │   └── medgemma_batch_analysis.py
│   └── visualization/
│
├── docs/                       # Documentation
│   ├── FINAL_ANALYSIS_SUMMARY.md
│   ├── MODEL_COMPARISON_REPORT.md
│   ├── COMPREHENSIVE_ANALYSIS_UPDATE.md
│   └── README.md
│
├── archived_versions/          # Old/deprecated code
│   ├── medgemma_launch.py
│   ├── test_llava_quick.py
│   └── test_question_phrasing_*.py
│
├── one-pixel-attack/          # Adversarial attack experiments
│   ├── onepixel_attack_main.py
│   ├── medgemma_evaluator.py
│   └── mimic_robustness_results/
│
└── data/                      # Data files
    └── (various CSV and data files)
```

### Key Files

#### Main Analysis Scripts
- `analyze_llava_matched_to_medgemma.py` - Direct comparison analysis
- `batch_analysis_100samples.py` - MedGemma batch analysis
- `llava_batch_analysis.py` - LLaVA batch analysis

#### Model Implementations
- `medgemma_launch_mimic_fixed.py` - MedGemma with attention extraction
- `llava_rad_visualizer.py` - LLaVA with visualization capabilities

#### Results
- `FINAL_COMPARISON_REPORT.md` - Comprehensive model comparison
- `results/*/statistics/` - Statistical analyses for each experiment

### Recent Updates (2025-08-26)

1. **Repository Reorganized**: Files moved to appropriate directories
2. **Completed Analyses**:
   - MedGemma accuracy: 74% (100 samples)
   - LLaVA accuracy: 62% (100 samples)
   - Question phrasing robustness tested (63 groups)
   - Attention pattern analysis completed

3. **Key Findings**:
   - MedGemma shows superior medical VQA performance
   - Both models sensitive to phrasing but maintain consistency
   - Small attention shifts correlate with answer changes

### Usage

#### Run MedGemma Analysis
```bash
cd analysis/
python batch_analysis_100samples.py
```

#### Run LLaVA Analysis
```bash
cd scripts/analysis/
python llava_batch_analysis.py
```

#### Compare Models
```bash
python analyze_llava_matched_to_medgemma.py
```

### Requirements
- PyTorch 2.0+
- Transformers 4.36+
- CUDA-capable GPU (80GB recommended)
- MIMIC-CXR dataset access

### Citation
If you use this code, please cite:
- MedGemma paper
- LLaVA paper
- MIMIC-CXR dataset