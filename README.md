# LVLM Interpretation - Medical Vision-Language Models on MIMIC-CXR

## Overview
This repository contains analysis and visualization tools for two medical vision-language models:
- **MedGemma** - Google's medical vision-language model
- **LLaVA-Rad** - Radiology-specific LLaVA model

Both models are evaluated on the **MIMIC-CXR dataset** (chest X-rays).

## Project Structure

### Model-Specific Visualizers
- `medgemma_app.py` - Main MedGemma application interface
- `medgemma_visualizer.py` - MedGemma attention visualization
- `medgemma_mimic_analyzer.py` - MedGemma analysis on MIMIC-CXR dataset
- `medgemma_relevancy_analyzer.py` - MedGemma relevancy analysis
- `llavarad_visualizer.py` - LLaVA-Rad attention visualization

### Comparison & Analysis Tools
- `attention_comparison_visualizer.py` - Compares attention patterns between models
- `phrasing_results_visualizer.py` - Analyzes impact of question phrasing
- `plot_generator_saved_data.py` - Generates plots from saved results
- `report_generator_comprehensive.py` - Creates comprehensive comparison reports

### Data Processing
- `results_processor.py` - Processes completed analysis results
- `results_summary_generator.py` - Generates summaries from full results

### Notebooks
- `notebooks_medgemma_attention_demo.ipynb` - MedGemma attention demonstration
- `notebooks_model_comparison.ipynb` - Model comparison analysis
- `notebooks_qwen_attention.ipynb` - Qwen model attention analysis

### Directory Structure

#### Core Directories
- `analysis/` - Analysis scripts for both models
  - `llavarad_attention_analyzer.py` - LLaVA-Rad attention analysis
  - `compare_llavarad_medgemma.py` - Direct model comparison
  - `answer_change_analyzer.py` - Analyzes answer changes
  - `attention_change_analyzer.py` - Attention pattern changes
  - `batch_analysis_100samples.py` - Batch processing
  - `faithfulness_validation.py` - Faithfulness metrics
  - `llava/` - LLaVA-specific analysis
  - `medgemma/` - MedGemma-specific analysis

- `models/` - Model implementations
  - `medgemma/` - MedGemma model files
  - `llava/` - LLaVA-Rad model files

- `results/` - All analysis outputs
  - `medgemma_attention_analysis/` - MedGemma attention results
  - `llava_rad_attention_analysis/` - LLaVA-Rad attention results
  - `llava_matched_attention_analysis/` - Matched comparison results
  - `mimic_medgemma_analysis/` - MIMIC-CXR specific results

#### Supporting Directories
- `data/` - MIMIC-CXR dataset files
- `docs/` - Documentation
- `one-pixel-attack/` - Robustness testing
- `reports/` - Generated reports
- `scripts/` - Utility scripts
- `archive/` - Archived files (logs, old scripts, temporary results)
- `archived_versions/` - Previous script versions

## Documentation
- `documentation_repository_structure.md` - Repository structure documentation
- `report_final_comparison.md` - Final comparison report

## Configuration
- `pyproject.toml` - Project dependencies and configuration
- `uv.lock` - Dependency lock file
- `LICENSE` - Project license

## Archive Structure
The `archive/` directory contains:
- `logs/` - All log files from previous runs
- `old_scripts/` - Deprecated or superseded scripts
- `test_scripts/` - Test and debug scripts
- `demo_files/` - Demo and presentation files
- `documentation/` - Old documentation files
- `temporary_results/` - Temporary result files
- `legacy_analysis/` - Legacy analysis files