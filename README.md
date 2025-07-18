# MedGemma-Interpret: Interpretability for Medical Vision-Language Models

An interpretability tool for MedGemma models, specifically adapted for chest X-ray analysis with transparent attention visualization.

## Overview

This repository implements attention visualization and interpretability techniques for Google's MedGemma-4B model, focusing on chest X-ray analysis. The work is based on the LVLM-Interpret framework and adapted specifically for medical imaging applications.

## Credits

This implementation is based on the original LVLM-Interpret work:

```bibtex
@article{stan2024lvlm,
  title={LVLM-Interpret: an interpretability tool for large vision-language models},
  author={Stan, Gabriela Ben Melech and Aflalo, Estelle and Rohekar, Raanan Yehezkel and Bhiwandiwalla, Anahita and Tseng, Shao-Yen and Olson, Matthew Lyle and Gurwicz, Yaniv and Wu, Chenfei and Duan, Nan and Lal, Vasudev},
  journal={arXiv preprint arXiv:2404.03118},
  year={2024}
}
```

## Key Adaptations for MedGemma

While LVLM-Interpret provides a general framework for vision-language model interpretability, this implementation includes several medical-specific adaptations:

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

## Features

- **Comprehensive Attention Visualization**: Multiple complementary visualization techniques to understand model focus
- **Token-Level Interpretability**: Analyze attention patterns for individual words in the generated report
- **Custom Question Support**: Ask specific clinical questions about the X-ray
- **Real-time Analysis**: Interactive Gradio interface for immediate feedback
- **High-Resolution Visualizations**: Enhanced contrast and overlay visualizations for clinical clarity

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medgemma-interpret.git
cd medgemma-interpret

# Install dependencies
pip install torch transformers gradio matplotlib opencv-python pillow numpy
```

## Usage

```bash
python medgemma_launch.py
```

The application will:
1. Load the MedGemma-4B model
2. Start a Gradio interface on http://0.0.0.0:7860
3. Provide both local and public URL access

## File Structure

- `medgemma_launch.py` - Main launcher script that loads the model and starts the interface
- `medgemma_attention_visualizer.py` - Core visualization and interface implementation
- `medgemma_relevancy_analyzer.py` - Relevancy computation algorithms adapted from LVLM-Interpret

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

## Example Use Cases

1. **General Analysis**: Upload a chest X-ray for comprehensive radiological assessment
2. **Specific Queries**: Ask targeted questions like "Is there any sign of pneumonia?"
3. **Teaching Tool**: Understand which image regions contribute to specific findings
4. **Model Debugging**: Verify the model is attending to clinically relevant regions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Transformers library
- Gradio for web interface

## Limitations

- Currently optimized specifically for chest X-rays
- Requires significant GPU memory for model and attention storage
- Best results with high-quality medical images

## Future Work

- Extend to other medical imaging modalities
- Add support for 3D medical images
- Implement additional interpretability methods
- Clinical validation studies

## License

This project follows the licensing terms of both LVLM-Interpret and MedGemma models. Please refer to their respective licenses for usage restrictions.

## Acknowledgments

Special thanks to:
- The LVLM-Interpret team for the foundational interpretability framework
- Google Research for the MedGemma model
- The medical AI community for continuous feedback and improvements