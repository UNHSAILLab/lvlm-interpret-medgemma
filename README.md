# MedGemma-Interpret: Interpretability for Medical Vision-Language Models

An interpretability tool for MedGemma models, specifically adapted for chest X-ray analysis with transparent attention visualization.

## Overview

This repository implements attention visualization and interpretability techniques for Google's MedGemma-4B model, focusing on chest X-ray analysis. The work is based on the LVLM-Interpret framework and adapted specifically for medical imaging applications using MedGemma Architecture.


## Key Adaptations for MedGemma

This implementation includes several medical-specific adaptations:

1. **Medical Domain Focus**: Specifically optimized for chest X-ray interpretation using MedGemma-4B
2. **Clinical Relevancy Maps**: Multiple attention visualization methods tailored for medical imaging:
   - Raw attention heatmaps
   - Layer-weighted relevancy 
   - Head importance analysis
   - Attention flow visualization
   - Consensus relevancy combining multiple methods

3. **Medical Report Generation**: Structured output format for radiological findings including:
   - Lung field analysis
   - Cardiac assessment
   - Bone abnormalities
   - Soft tissue findings
   - Medical device identification

4. **Interactive Token Analysis**: Click on any word in the generated report to see which regions of the X-ray the model attended to when generating that specific token


## Usage

```bash
python medgemma_launch.py
```

The application will:
1. Load the MedGemma-4B model
2. Start a Gradio interface on http://0.0.0.0:7860
3. Provide both local and public URL access


## Technical Details

### Attention Extraction
The tool extracts attention weights from all layers of MedGemma and provides multiple aggregation strategies:
- Mean attention across heads
- Weighted by head importance scores
- Attention flow computation through layers
- Consensus scoring combining multiple methods

### Visualization Methods
1. **Raw Attention**: Direct attention weights from the last layer
2. **Layer-Weighted Relevancy**: Aggregates attention across all layers with learned weights
3. **Head Importance**: Weights attention by the importance of each attention head
4. **Attention Flow**: Traces attention propagation through the network
5. **Consensus Map**: Combines multiple methods for robust visualization

### Medical Adaptations
- Optimized for grayscale medical images
- High-contrast visualization for subtle findings
- Contour detection for region-of-interest identification
- Medical terminology in generated reports


## Credits

This implementation is an adaptation of original LVLM-Interpret work but rewired to work with SigLip vision tower:

```bibtex
@article{stan2024lvlm,
  title={LVLM-Interpret: an interpretability tool for large vision-language models},
  author={Stan, Gabriela Ben Melech and Aflalo, Estelle and Rohekar, Raanan Yehezkel and Bhiwandiwalla, Anahita and Tseng, Shao-Yen and Olson, Matthew Lyle and Gurwicz, Yaniv and Wu, Chenfei and Duan, Nan and Lal, Vasudev},
  journal={arXiv preprint arXiv:2404.03118},
  year={2024}
}
```
