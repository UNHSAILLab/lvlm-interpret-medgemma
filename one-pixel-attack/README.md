# One-Pixel Attack Robustness Evaluation

This module implements one-pixel adversarial attacks on chest X-ray images to evaluate the robustness of the MedGemma model.

## Overview

The evaluation consists of three main components:
1. **Attack Generation**: Creates adversarial examples by modifying a single pixel within lung regions
2. **Model Evaluation**: Tests MedGemma on both original and adversarial images
3. **Robustness Analysis**: Compares model performance to measure robustness

## Quick Start

```bash
# Run complete evaluation pipeline
python run_robustness_evaluation.py \
    --data-path /path/to/questions.csv \
    --source-folder /path/to/original/images \
    --output-folder /path/to/adversarial/images \
    --results-folder ./results
```

## Features

### Attack Generation
- Differential evolution optimization to find optimal pixel perturbation
- Constraints to lung regions for medical relevance
- Configurable attack parameters (iterations, population size)
- Detailed logging of attack success metrics

### Model Evaluation
- Structured evaluation framework for MedGemma
- Support for multiple image views (PA, lateral)
- JSON response parsing with error handling
- Progress tracking and intermediate saves

### Robustness Analysis
- Accuracy comparison (baseline vs adversarial)
- Attack success rate metrics
- Confidence score analysis
- Per-question vulnerability assessment

## Usage Examples

### Basic Evaluation
```bash
python run_robustness_evaluation.py \
    --data-path questions.csv \
    --source-folder /data/chest_xrays \
    --output-folder /data/adversarial \
    --limit 100
```

### Resume from Previous Run
```bash
python run_robustness_evaluation.py \
    --data-path questions.csv \
    --source-folder /data/chest_xrays \
    --output-folder /data/adversarial \
    --skip-attack-generation \
    --resume-baseline results/run_20240115/baseline_results.csv \
    --resume-attack results/run_20240115/onepixel_results.csv
```

### Custom Attack Parameters
```bash
python run_robustness_evaluation.py \
    --data-path questions.csv \
    --source-folder /data/chest_xrays \
    --output-folder /data/adversarial \
    --max-iter 200 \
    --pop-size 300 \
    --pathology Pneumonia
```

## Output Structure

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── attack_generation_summary.csv   # Attack success metrics
    ├── baseline_results.csv            # Original image evaluations
    ├── onepixel_results.csv           # Adversarial image evaluations
    ├── robustness_summary.json        # Overall robustness metrics
    ├── per_question_analysis.csv      # Question-level analysis
    └── detailed_comparison.csv        # Sample-by-sample comparison
```

## Input Data Format

The input CSV should contain:
- `study_id`: Unique identifier
- `question`: The question text
- `options`: Answer choices
- `correct_answer`: Correct answer (A/B/C/D)
- `image_path`: Path(s) to image files

## Key Metrics

- **Baseline Accuracy**: Model accuracy on original images
- **Attack Accuracy**: Model accuracy on adversarial images
- **Attack Success Rate**: Percentage of predictions changed
- **Targeted Success Rate**: Percentage of correct predictions made incorrect
- **Confidence Drop**: Average decrease in model confidence

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- torchxrayvision
- scikit-image
- pandas, numpy, PIL

## Notes

- Attacks are constrained to lung regions for medical validity
- The surrogate model (DenseNet) is used for efficiency
- Results include detailed logs for reproducibility
- Supports batch processing with resume capability