# MedGemma 4B Multimodal VLM - MIMIC-CXR Analysis Platform

## Quick Start

Run the application with:
```bash
python medgemma_app.py
```

Or use the full filename:
```bash
python medgemma_launch_mimic_fixed.py
```

## Options

- `--gpu N`: Select specific GPU (default: auto-select)
- `--min-memory N`: Minimum free GPU memory in GB (default: 15)
- `--port N`: Server port (default: 7860)
- `--host HOST`: Server host (default: 0.0.0.0)

## Examples

```bash
# Auto-select GPU with most free memory
python medgemma_app.py

# Use specific GPU
python medgemma_app.py --gpu 0

# Custom port
python medgemma_app.py --port 8080
```

## Features

- **MIMIC-CXR Question Analysis**: Analyze chest X-rays with predefined medical questions
- **Token-Conditioned Attention**: Visualize attention for specific tokens
- **Prompt Sensitivity Analysis**: Compare different prompt variations
- **Ground Truth Comparison**: All tabs show ground truth answers
- **Robust Error Handling**: Graceful fallbacks for edge cases

## Archived Versions

Previous versions are stored in `archived_versions/` for reference.

---
Developed by SAIL Lab at University of New Haven