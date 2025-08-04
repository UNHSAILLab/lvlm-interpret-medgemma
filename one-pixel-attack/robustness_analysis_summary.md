# One-Pixel Attack Robustness Analysis Summary

## Executive Summary

Our analysis reveals a counterintuitive finding: **the one-pixel adversarial attack actually improves MedGemma's performance** on pneumonia detection tasks. The model accuracy increased from 65% (baseline) to 85% (after attack), with the improvement entirely attributable to the preprocessing pipeline rather than the pixel modification itself.

## Key Findings

### Performance Metrics (20 samples)
- **Baseline Accuracy**: 65.0%
- **Preprocessed Only Accuracy**: 85.0% 
- **One-Pixel Attack Accuracy**: 85.0%
- **Preprocessing Effect**: +20.0% (improvement from baseline to preprocessed)
- **Attack Effect**: 0.0% (no change from preprocessed to attack)
- **Attack Success Rate**: 20.0% (predictions that changed)
- **Targeted Success Rate**: 0.0% (no correct predictions made incorrect)

### Critical Insights
1. **All improvement comes from preprocessing**: The 20% accuracy gain occurs when images are processed through the attack pipeline WITHOUT any pixel modification
2. **The pixel attack itself has no effect**: Comparing preprocessed-only vs one-pixel attack shows 0% difference
3. **No successful targeted attacks**: The attack failed to convert any correct predictions to incorrect ones
4. **Confidence scores remain stable**: Average confidence ~94.5% across all conditions

## Technical Implementation Details

### One-Pixel Attack Algorithm

The implementation follows a sophisticated multi-resolution approach:

1. **Original Resolution Preservation**
   - Loads chest X-rays at native resolution
   - Maintains original image quality throughout

2. **Surrogate Model Optimization** 
   - Resizes to 224×224 for DenseNet surrogate model
   - Uses differential evolution to find optimal attack:
     - Pixel location (x, y) constrained to lung regions
     - Perturbation value δ ∈ [-255, 255]
   - Lung mask extraction via adaptive thresholding

3. **Attack Application**
   - Scales coordinates from 224×224 back to original dimensions
   - Modifies single pixel in original resolution image
   - Saves at original resolution with only 1 pixel changed

### Three-Way Evaluation Framework

To isolate effects, we implemented three evaluation modes:

```python
# 1. Baseline: Original images from source folder
baseline_results = evaluator.evaluate_dataset(
    data=data,
    folder_path=source_folder,
    attack_type='baseline'
)

# 2. Preprocessed: Exact copies in new folder (control)
preprocessed_results = evaluator.evaluate_dataset(
    data=data,
    folder_path=preprocessed_folder,  
    attack_type='preprocessed'
)

# 3. Attack: Images with one pixel modified
attack_results = evaluator.evaluate_dataset(
    data=data,
    folder_path=attack_folder,
    attack_type='onepixel'
)
```

### Image Processing Pipeline

The preprocessing function that shows the improvement:

```python
def _load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(image_path)
    
    # Handle 16-bit DICOM images
    if img.mode == 'I;16':
        img_array = np.array(img)
        img_min, img_max = img_array.min(), img_array.max()
        img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_8bit = np.array(img.convert('L'))
    
    return img_8bit, img_8bit.shape
```

## Hypothesis for Performance Improvement

The unexpected accuracy improvement likely stems from:

1. **Image Normalization**: Converting 16-bit medical images to 8-bit may reduce noise and improve model performance
2. **Consistent Processing**: The pipeline ensures all images go through identical preprocessing
3. **File Format Standardization**: Converting various formats to consistent PNG representation

## Recommendations for Further Investigation

1. **Analyze Image Differences**
   ```python
   # Compare pixel distributions
   original_img = load_original_image(path)
   processed_img = load_processed_image(path)
   
   # Check bit depth, normalization, format conversions
   print(f"Original: {original_img.dtype}, range: [{original_img.min()}, {original_img.max()}]")
   print(f"Processed: {processed_img.dtype}, range: [{processed_img.min()}, {processed_img.max()}]")
   ```

2. **Test Preprocessing Components Individually**
   - Bit depth conversion only
   - Format conversion only  
   - Full pipeline without attack

3. **Validate on Larger Dataset**
   - Current results based on 20 samples
   - Recommend testing on full dataset

4. **Investigate Model Sensitivity**
   - Why does MedGemma perform better on preprocessed images?
   - Is the model trained on similar preprocessing?

## Code to Continue Analysis

```python
# Load the detailed results for deeper analysis
import pandas as pd
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("./robustness_results/run_YYYYMMDD_HHMMSS/")
baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
preprocessed_df = pd.read_csv(results_dir / "preprocessed_results.csv") 
attack_df = pd.read_csv(results_dir / "onepixel_results.csv")
detailed_comparison = pd.read_csv(results_dir / "detailed_comparison.csv")

# Analyze which questions improved
improved_mask = (
    (detailed_comparison['model_answer_baseline'] != detailed_comparison['correct_answer_baseline']) &
    (detailed_comparison['model_answer_preprocessed'] == detailed_comparison['correct_answer_preprocessed'])
)
improved_questions = detailed_comparison[improved_mask]

print(f"Questions that improved with preprocessing: {len(improved_questions)}")
print(improved_questions[['study_id', 'question_baseline', 'model_answer_baseline', 'model_answer_preprocessed']])
```

## Prompt for Continuing Investigation

To continue this analysis in a new session:

> "I'm investigating an unexpected result where a one-pixel adversarial attack on chest X-rays actually improves MedGemma's pneumonia detection accuracy from 65% to 85%. Through controlled experiments with three conditions (baseline, preprocessed-only, and one-pixel attack), we found that ALL improvement comes from the image preprocessing pipeline, not the attack itself. The preprocessing involves loading images with PIL, converting 16-bit to 8-bit if needed, and saving as PNG. The model shows 0% difference between preprocessed-only and attacked images. I need help understanding why this preprocessing improves model performance and how to further investigate this phenomenon. The detailed results show targeted attack success rate of 0%, meaning no correct predictions became incorrect."

## Technical Details for Reference

- **Model**: MedGemma (medical vision-language model)
- **Surrogate Model**: DenseNet-121 trained on chest X-rays
- **Attack Constraint**: Lung regions only (via adaptive thresholding)
- **Optimization**: Differential evolution (100 iterations, population 200)
- **Image Processing**: Original resolution preserved, 224×224 only for attack search
- **Evaluation**: VQA-style pneumonia detection questions